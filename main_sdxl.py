#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import functools
import gc
import logging
import math
import os
import random
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import accelerate
import diffusers
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import ConcatDataset
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AnimateDiffPipeline,
    AnimateDiffSDXLPipeline,
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    LCMScheduler,
    MotionAdapter,
    StableDiffusionPipeline,
    TextToVideoSDPipeline,
    UNet2DConditionModel,
    UNet3DConditionModel,
    UNetMotionModel,
)
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, export_to_video, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
from tabulate import tabulate
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize, RandomCrop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig, CLIPTextModelWithProjection, CLIPTokenizer

from args import parse_args
from dataset.webvid_dataset_wbd import Text2VideoDataset
from dataset.opensora_dataset import OpenSoraPlan, LivephotoDataset
from models.discriminator_handcraft import (
    ProjectedDiscriminator,
    get_dino_features,
    preprocess_dino_input,
)
from models.spatial_head import IdentitySpatialHead, SpatialHead
from utils.diffusion_misc import *
from utils.dist import dist_init, dist_init_wo_accelerate, get_deepspeed_config
from utils.image_util import fit_image_to_size
from utils.misc import *
from utils.wandb import setup_wandb
from utils.vis import ImageRealtimeVis
import imageio
from torchvision import transforms
from omegaconf import OmegaConf

MAX_SEQ_LENGTH = 77

if is_wandb_available():
    import wandb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(name)s] - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)

def save_args_to_yaml(args, yaml_path):

    args_dict = vars(args)

    conf = OmegaConf.create(args_dict)

    OmegaConf.save(conf, yaml_path)


def save_to_local(save_dir: str, prompt: str, video, fps: int=10):
    if len(prompt) > 256:
        prompt = prompt[:256]
    prompt = prompt.replace(" ", "_")
    logger.info(f"Saving images to {save_dir}")

    imageio.mimwrite(os.path.join(save_dir, f"{prompt}.mp4"), video, fps=fps)
    # export_to_video(video, os.path.join(save_dir, f"{prompt}.mp4"))


class STDBank:

    def __init__(
        self,
        bank_len: int=4,
        last_timesteps_index: int=1,
        return_mode: str='t',
    ):
        self.bank_len = bank_len
        self.last_timesteps_index = last_timesteps_index
        self.bank = [[] for _ in range(self.bank_len)]
        self.random_func = np.random.RandomState()
        self.return_mode = return_mode
    
    def push(self, timesteps_index, model_pred, encoded_text, text, pixel_values, idx=None):

        buffer = dict(
            timesteps_index=timesteps_index,
            model_pred=model_pred,
            encoded_text=encoded_text,
            text=text,
            pixel_values=pixel_values,
        )

        if idx is not None:
            if self.return_mode == 't':
                self.bank[idx][0] = buffer
            else:
                self.bank[idx].append(buffer)
        else:
            empty_bank_idx = self._get_empty_bank_idx()
            if empty_bank_idx < 0:
                return 
            self.bank[empty_bank_idx].append(buffer)
    
    def pop(self, idx):
        self.bank[idx] = []
    
    def _get_empty_bank_idx(self):

        for i in range(self.bank_len):
            if len(self.bank[i]) == 0:
                return i
        return -1
    
    @property
    def current_bank_len(self):
        count = 0
        for i in range(self.bank_len):
            if len(self.bank[i]) > 0:
                count += 1
        return count
    
    def update(self, timesteps_index, model_pred, encoded_text, text, pixel_values, idx=None):

        if timesteps_index <= self.last_timesteps_index:
            if idx is not None:
                self.pop(idx)
                return
            else:
                raise ValueError(f"when timestep == last_timestep, idx should not be None.")
        
        self.push(timesteps_index, model_pred, encoded_text, text, pixel_values, idx)

    def get_ramdom_item(self):

        if self.current_bank_len > 0:
            selected_idx = self.random_func.randint(0, self.current_bank_len)
            sub_bank_len = len(self.bank[selected_idx])

            if self.return_mode == 't':
                if sub_bank_len > 0:
                    t_idx = sub_bank_len - 1
                    return selected_idx, self.bank[selected_idx][t_idx]

            elif self.return_mode == 't_it':
                if sub_bank_len > 1:
                    t_idx, it_idx = self.random_func.randint(0, sub_bank_len), sub_bank_len - 1
                    return selected_idx, (
                        self.bank[selected_idx][t_idx], 
                        self.bank[selected_idx][it_idx]
                    )

            elif self.return_mode == 't_it_s':
                if sub_bank_len > 2:
                    t_idx, s_idx = self.random_func.randint(0, sub_bank_len - 1), sub_bank_len - 1
                    it_idx = self.random_func.randint(t_idx, s_idx)
                    return selected_idx, (
                        self.bank[selected_idx][t_idx], 
                        self.bank[selected_idx][it_idx],
                        self.bank[selected_idx][s_idx],
                    )
        return None, None
    
    def debug_print(self):

        print("+++++++++++++++ STD DEBUG +++++++++++++++")
        print(f"current bank len: {self.current_bank_len}")
        for i in range(self.bank_len):
            print(f"{i}th timestep index: {[int(self.bank[i][j]['timesteps_index']) for j in range(len(self.bank[i]))]}")
        print("+++++++++++++++ STD DEBUG +++++++++++++++")

def main(args):
    # torch.multiprocessing.set_sharing_strategy("file_system")
    dist_init()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    setup_wandb()

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
        # deepspeed_plugin=deepspeed_plugin,
    )

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    # Make one log on every process with the configuration for debugging.
    logger.info("Printing accelerate state", main_process_only=False)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_batch_size / 128
        args.disc_learning_rate = (
            args.disc_learning_rate * total_batch_size * args.disc_tsn_num_frames / 128
        )
        logger.info(f"Scaling learning rate to {args.learning_rate}")
        logger.info(f"Scaling discriminator learning rate to {args.disc_learning_rate}")

    sorted_args = sorted(vars(args).items())
    logger.info(
        "\n" + tabulate(sorted_args, headers=["key", "value"], tablefmt="rounded_grid"),
        main_process_only=True,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
                private=True,
            ).repo_id

    try:
        accelerator.wait_for_everyone()
    except Exception as e:
        logger.error(f"Failed to wait for everyone: {e}")
        dist_init_wo_accelerate()
        accelerator.wait_for_everyone()

    # 1. Create the noise scheduler and the desired noise schedule.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="scheduler",
        revision=args.teacher_revision,
        rescale_betas_zero_snr=True if args.zero_snr else False,
        beta_schedule=args.beta_schedule,
    )
    if args.zero_snr:
        noise_scheduler.config.prediction_type = 'v_prediction'

    # DDPMScheduler calculates the alpha and sigma noise schedules (based on the alpha bars) for us
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    # Initialize the DDIM ODE solver for distillation.
    solver = DDIMSolverV2(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=args.num_ddim_timesteps,
    )

    # 2. Load tokenizers from SD 1.X/2.X checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="tokenizer",
        revision=args.teacher_revision,
        use_fast=False,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(args.pretrained_teacher_model, subfolder="tokenizer_2")

    # 3. Load text encoders from SD 1.X/2.X checkpoint.
    # import correct text encoder classes
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="text_encoder",
        revision=args.teacher_revision,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.pretrained_teacher_model, subfolder="text_encoder_2")

    # 4. Load VAE from SD 1.X/2.X checkpoint
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="vae",
        revision=args.teacher_revision,
    )

    # 5. Load teacher U-Net from SD 1.X/2.X checkpoint
    teacher_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="unet",
        revision=args.teacher_revision,
    )
    if args.use_motion:
        teacher_motion_adapter = MotionAdapter.from_pretrained(args.motion_adapter_path)
        teacher_unet = UNetMotionModel.from_unet2d(teacher_unet, teacher_motion_adapter)

    # 5.1 Load DINO
    dino = torch.hub.load(
        "/mnt/nj-public02/usr/xusijie/github/dinov2",
        "dinov2_vits14",
        pretrained=False,
        source='local',
    )
    ckpt_path = "weights/dinov2_vits14_pretrain.pth"
    state_dict = torch.load(ckpt_path, map_location="cpu")
    dino.load_state_dict(state_dict)
    logger.info(f"Loaded DINO model from {ckpt_path}")
    dino.eval()

    # 5.2 Load sentence-level CLIP
    open_clip_model, *_ = open_clip.create_model_and_transforms(
        "ViT-g-14",
        pretrained="weights/open_clip_pytorch_model.bin",
    )
    open_clip_tokenizer = open_clip.get_tokenizer("ViT-g-14")

    # 6. Freeze teacher vae, text_encoder, and teacher_unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    teacher_unet.requires_grad_(False)
    dino.requires_grad_(False)
    open_clip_model.requires_grad_(False)
    normalize_fn = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    # 7. Create online student U-Net.
    # For whole model fine-tuning, this will be updated by the optimizer (e.g.,
    # via backpropagation.)
    # Add `time_cond_proj_dim` to the student U-Net if `teacher_unet.config.time_cond_proj_dim` is None
    unet = deepcopy(teacher_unet)
    unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if args.use_motion:
        motion_adapter = MotionAdapter.from_pretrained(args.motion_adapter_path)
        unet = UNetMotionModel.from_unet2d(unet, motion_adapter)

    if args.cd_target in ["learn", "hlearn"] and args.use_motion:
        if args.cd_target == "learn":
            spatial_head = SpatialHead(num_channels=4, num_layers=2, kernel_size=1)
            target_spatial_head = SpatialHead(
                num_channels=4, num_layers=2, kernel_size=1
            )
            logger.info("Using SpatialHead for spatial head")
        elif args.cd_target == "hlearn":
            spatial_head = SpatialHead(num_channels=4, num_layers=5, kernel_size=3)
            target_spatial_head = SpatialHead(
                num_channels=4, num_layers=5, kernel_size=3
            )
            logger.info("Using SpatialHead for spatial head")
        else:
            raise ValueError(f"cd_target {args.cd_target} is not supported.")

        spatial_head.train()
        target_spatial_head.load_state_dict(spatial_head.state_dict())
        target_spatial_head.train()
        target_spatial_head.requires_grad_(False)
    else:
        spatial_head = None
        target_spatial_head = None

    unet.train()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # 8. Add LoRA to the student U-Net, only the LoRA projection matrix will be updated by the optimizer.
    if args.lora_target_modules is not None:
        logger.warning(
            "We are currently ignoring the `lora_target_modules` argument. As of now, LoRa does not support Conv3D layers."
        )
        lora_target_modules = [
            module_key.strip() for module_key in args.lora_target_modules.split(",")
        ]
    else:
        lora_target_modules = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "conv1",
            "conv2",
            "conv_shortcut",
            "downsamplers.0.conv",
            "upsamplers.0.conv",
            "time_emb_proj",
        ]

    # Currently LoRA does not support Conv3D, thus removing the Conv3D
    # layers from the list of target modules.
    key_list = []
    for name, module in unet.named_modules():
        if any([name.endswith(module_key) for module_key in lora_target_modules]):
            if args.remove_motion_lora is True and 'motion' in name:
                continue
            key_list.append(name)

    lora_config = LoraConfig(
        r=args.lora_rank,
        target_modules=key_list,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    unet = get_peft_model(unet, lora_config)

    if (
        args.from_pretrained_unet is not None
        and args.from_pretrained_unet != "None"
    ):
        # TODO currently only supports LoRA
        logger.info(f"Loading pretrained UNet from {args.from_pretrained_unet}")
        unet.load_adapter(
            args.from_pretrained_unet,
            "default",
            is_trainable=True,
            torch_device="cpu",
        )
    unet.print_trainable_parameters()

    # 8.1. Create discriminator for the student U-Net.
    c_dim = 1024
    discriminator = ProjectedDiscriminator(
        embed_dim=dino.embed_dim, c_dim=c_dim
    )  # TODO add dino name and patch size
    if args.from_pretrained_disc is not None and args.from_pretrained_disc != "None":
        try:
            disc_state_dict = load_file(
                os.path.join(
                    args.from_pretrained_disc,
                    "discriminator",
                    "diffusion_pytorch_model.safetensors",
                )
            )
            discriminator.load_state_dict(disc_state_dict)
            logger.info(
                f"Loaded pretrained discriminator from {args.from_pretrained_disc}"
            )
        except Exception as e:
            logger.error(f"Failed to load pretrained discriminator: {e}")
    discriminator.train()

    # 9. Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device)
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    dino.to(accelerator.device, dtype=weight_dtype)
    open_clip_model.to(accelerator.device)

    # Move teacher_unet to device, optionally cast to weight_dtype
    teacher_unet.to(accelerator.device)
    if args.cast_teacher_unet:
        teacher_unet.to(dtype=weight_dtype)
    if args.cd_target in ["learn", "hlearn"] and args.use_motion:
        target_spatial_head.to(accelerator.device)

    # Also move the alpha and sigma noise schedules to accelerator.device.
    alpha_schedule = alpha_schedule.to(accelerator.device)
    sigma_schedule = sigma_schedule.to(accelerator.device)
    # Move the ODE solver to accelerator.device.
    solver = solver.to(accelerator.device)

    # 10. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                unet_ = accelerator.unwrap_model(unet)
                lora_state_dict = get_peft_model_state_dict(
                    unet_, adapter_name="default"
                )
                # update lora keys
                lora_keys = list(lora_state_dict.keys())
                for key in lora_keys:
                    new_key = key.replace('base_model.model.', '')
                    lora_state_dict[new_key] = lora_state_dict[key]
                    del lora_state_dict[key]
                StableDiffusionPipeline.save_lora_weights(
                    os.path.join(output_dir, "unet_lora"), lora_state_dict
                )
                # save weights in peft format to be able to load them back
                unet_.save_pretrained(output_dir)

                discriminator_ = accelerator.unwrap_model(discriminator)
                discriminator_.save_pretrained(
                    os.path.join(output_dir, "discriminator")
                )

                if args.cd_target in ["learn", "hlearn"] and args.use_motion:
                    spatial_head_ = accelerator.unwrap_model(spatial_head)
                    spatial_head_.save_pretrained(
                        os.path.join(output_dir, "spatial_head")
                    )
                    target_spatial_head_ = accelerator.unwrap_model(
                        target_spatial_head
                    )
                    target_spatial_head_.save_pretrained(
                        os.path.join(output_dir, "target_spatial_head")
                    )

                for _, model in enumerate(models):
                    # make sure to pop weight so that corresponding model is not saved again
                    if len(weights) > 0:
                        weights.pop()

        def load_model_hook(models, input_dir):
            # load the LoRA into the model
            unet_ = accelerator.unwrap_model(unet)
            unet_.load_adapter(
                input_dir, "default", is_trainable=True, torch_device="cpu"
            )

            disc_state_dict = load_file(
                os.path.join(
                    input_dir,
                    "discriminator",
                    "diffusion_pytorch_model.safetensors",
                )
            )
            disc_ = accelerator.unwrap_model(discriminator)
            disc_.load_state_dict(disc_state_dict)
            del disc_state_dict

            if args.cd_target in ["learn", "hlearn"] and args.use_motion:
                spatial_head_state_dict = load_file(
                    os.path.join(
                        input_dir,
                        "spatial_head",
                        "diffusion_pytorch_model.safetensors",
                    )
                )
                spatial_head_ = accelerator.unwrap_model(spatial_head)
                spatial_head_.load_state_dict(spatial_head_state_dict)
                del spatial_head_state_dict
                target_spatial_head_state_dict = load_file(
                    os.path.join(
                        input_dir,
                        "target_spatial_head",
                        "diffusion_pytorch_model.safetensors",
                    )
                )
                target_spatial_head_ = accelerator.unwrap_model(target_spatial_head)
                target_spatial_head_.load_state_dict(target_spatial_head_state_dict)
                del target_spatial_head_state_dict

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                models.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # 12. Optimizer creation
    if args.cd_target in ["learn", "hlearn"] and args.use_motion:
        unet_params = list(unet.parameters()) + list(spatial_head.parameters())
    else:
        unet_params = unet.parameters()

    optimizer = optimizer_class(
        unet_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    disc_optimizer = optimizer_class(
        discriminator.parameters(),
        lr=args.disc_learning_rate,
        betas=(args.disc_adam_beta1, args.disc_adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 13. Dataset creation and data processing
    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(
        prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True
    ):
        prompt_embeds = encode_prompt(
            prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train
        )
        return prompt_embeds
    
    def compute_embeddings_sdxl(
        batch, tokenizer, tokenizer_two, text_encoder, text_encoder_two, device,
    ):
    # Get the text embedding for conditioning
        with torch.no_grad():
            encoder_hidden_states, pooled_prompt_embeds = encode_prompt_sdxl(
                batch['text'],
                tokenizer=tokenizer,
                tokenizer2=tokenizer_two,
                text_encoder=text_encoder,
                text_encoder2=text_encoder_two,
                device=device,
            )
            add_text_embeds = pooled_prompt_embeds
            add_time_ids = torch.cat([
                get_add_time_ids(
                    original_size = original_size,
                    crops_coords_top_left = crops_coords_top_left,
                    target_size = target_size,
                    dtype = weight_dtype
                ) for original_size, crops_coords_top_left, target_size in zip(batch['original_size'], batch['crop_top_left'], batch['target_size'])
            ], dim=0)

            encoder_hidden_states = encoder_hidden_states.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device)

            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return dict(prompt_embeds=encoder_hidden_states, **added_cond_kwargs)

    def preprocess_train(images):
        # image aug
        train_resize_crop = lambda x: fit_image_to_size(x, args.resolution, args.resolution)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Normalize([0.5], [0.5])

        original_size = images.shape[-2:]
        if random.random() < 0.5:
            images = train_flip(images)
        images, crop_top_left, target_size = train_resize_crop(images)
        images = train_transforms(images)
        return images, original_size, crop_top_left, target_size
    
    def collate_fn(batch):
        video = torch.stack([x['video'] for x in batch])
        texts = [x['text'] for x in batch]
        __key__ = [x['__key__'] for x in batch]
        original_sizes = [x['original_size'] for x in batch]
        crop_top_lefts = [x['crop_top_left'] for x in batch]
        target_sizes = [x['target_size'] for x in batch]

        result = dict(
            video=video,
            text=texts,
            __key__=__key__,
            original_size=original_sizes,
            crop_top_left=crop_top_lefts,
            target_size=target_sizes,
        )
        return result
    

    dataset_soraplan = OpenSoraPlan(
        sample_size=args.resolution,
        sample_stride=args.frame_interval, 
        sample_n_frames=args.num_frames,
        is_image=False if args.use_motion else True,
        process_fn=preprocess_train,
    )
    dataset_livephoto = LivephotoDataset(
        sample_size=args.resolution,
        sample_stride=args.frame_interval, 
        sample_n_frames=args.num_frames,
        is_image=False if args.use_motion else True,
        process_fn=preprocess_train,
    )
    dataset = ConcatDataset([dataset_soraplan, dataset_livephoto])
    train_dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.train_batch_size, 
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )

    compute_embeddings_fn = functools.partial(
        compute_embeddings_sdxl,
        text_encoder=text_encoder,
        text_encoder_two=text_encoder_two,
        tokenizer=tokenizer,
        tokenizer_two=tokenizer_two,
        device=accelerator.device,
    )

    # 14. LR Scheduler creation
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_batches = math.ceil(len(train_dataloader) / (args.train_batch_size * accelerator.num_processes))
    num_update_steps_per_epoch = math.ceil(
        num_batches / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    disc_lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=disc_optimizer,
        num_warmup_steps=args.lr_warmup_steps if args.disc_start_step == 0 else 0,
        num_training_steps=args.max_train_steps - args.disc_start_step,
    )

    # 15. Prepare for training
    # Prepare everything with our `accelerator`.
    if args.cd_target in ["learn", "hlearn"] and args.use_motion:
        (
            unet,
            spatial_head,
            discriminator,
            optimizer,
            disc_optimizer,
            lr_scheduler,
            disc_lr_scheduler,
        ) = accelerator.prepare(
            unet,
            spatial_head,
            discriminator,
            optimizer,
            disc_optimizer,
            lr_scheduler,
            disc_lr_scheduler,
        )
    else:
        (
            unet,
            discriminator,
            optimizer,
            disc_optimizer,
            lr_scheduler,
            disc_lr_scheduler,
        ) = accelerator.prepare(
            unet,
            discriminator,
            optimizer,
            disc_optimizer,
            lr_scheduler,
            disc_lr_scheduler,
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        num_batches / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # remove list objects to avoid bug in tensorboard
        tracker_config = {
            k: v for k, v in vars(args).items() if not isinstance(v, list)
        }
        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
            init_kwargs={"wandb": {"name": args.tracker_run_name}},
        )

    with torch.no_grad():
        uncond_prompt_embeds, _ = encode_prompt_sdxl(
            [""] * args.train_batch_size,
            tokenizer=tokenizer,
            tokenizer2=tokenizer_two,
            text_encoder=text_encoder,
            text_encoder2=text_encoder_two,
            device=accelerator.device,
        )

    # 16. Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {num_batches}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(
        f"  Num learnable parameters = {sum([p.numel() for p in unet.parameters() if p.requires_grad]) / 1e6} M"
    )
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [
                d
                for d in dirs
                if (d.startswith("checkpoint") and "step" not in d and "final" not in d)
            ]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if os.path.exists(os.path.join(args.output_dir, path)):
                accelerator.load_state(os.path.join(args.output_dir, path))
            else:
                accelerator.load_state(args.resume_from_checkpoint)
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    gc.collect()
    torch.cuda.empty_cache()

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    last_update_r1_step = global_step

    # STD: init std bank
    if args.use_std:
        std_bank = STDBank()
    
    # monitor
    if accelerator.is_main_process and args.use_monitor is True:
        vis_dir = f"{args.output_dir}/vis"
        os.makedirs(vis_dir, exist_ok=True)
        train_monitor = ImageRealtimeVis(save_dir=vis_dir, save_num=args.monitor_num, save_keys=["batch_tensor"])

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet), accelerator.accumulate(discriminator):
                # 1. Load and process the image and text conditioning
                video, text = batch["video"], batch["text"]

                video = video.to(accelerator.device, non_blocking=True)
                encoded_text = compute_embeddings_fn(batch)

                pixel_values = video.to(dtype=weight_dtype)
                if vae.dtype != weight_dtype:
                    vae.to(dtype=weight_dtype)

                # encode pixel values with batch size of at most args.vae_encode_batch_size
                latents = encode_and_sample(
                    pixel_values, 
                    vae, 
                    args.num_frames, 
                    args.use_motion, 
                    weight_dtype,
                )
                bsz = latents.shape[0]

                # STD: get random item
                if random.random() < args.std_rate and args.use_std:
                    selected_idx, std_item = std_bank.get_ramdom_item()
                else:
                    selected_idx, std_item = None, None
                if std_item is not None:
                    index = std_item['timesteps_index'].to(latents.device)
                    noisy_model_input = std_item['model_pred'].to(latents.device).to(weight_dtype)

                # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
                # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
                topk = (
                    noise_scheduler.config.num_train_timesteps
                    // args.num_ddim_timesteps
                )
                if std_item is None:
                    if args.use_std:
                        index = torch.Tensor([math.ceil(args.num_ddim_timesteps * args.strength)]).to(latents.device).long()
                    else:
                        index = torch.randint(
                            1, math.ceil(args.num_ddim_timesteps * args.strength), (bsz,), device=latents.device
                        ).long()
                start_timesteps = solver.ddim_timesteps[index]
                timesteps = start_timesteps - topk
                timesteps = torch.where(
                    timesteps < 0, torch.zeros_like(timesteps), timesteps
                )

                inference_indices = np.linspace(
                    0, len(solver.ddim_timesteps) * args.strength, num=args.multiphase, endpoint=False
                )
                inference_indices = np.floor(inference_indices).astype(np.int64)
                inference_indices = (
                    torch.from_numpy(inference_indices).long().to(timesteps.device)
                )

                # 4. Sample a random guidance scale w from U[w_min, w_max] and embed it
                # Note that for LCM-LoRA distillation it is not necessary to use a guidance scale embedding
                w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                w = w.reshape([bsz] + [1] * (latents.dim() - 1))
                w = w.to(device=latents.device, dtype=latents.dtype)

                # if use predicted x_0, use the caption from the disc gt dataset
                # instead of from WebVid
                use_pred_x0 = False
                if global_step >= args.disc_start_step and not args.no_disc:
                    if args.cd_pred_x0_portion >= 0:
                        use_pred_x0 = random.random() < args.cd_pred_x0_portion

                    if args.disc_gt_data == "webvid":
                        pass
                    else:
                        gt_sample, gt_sample_caption = next(disc_gt_dataloader)
                        if use_pred_x0 and args.disc_same_caption:
                            text = gt_sample_caption
                            encoded_text = compute_embeddings_fn(text)

                # get CLIP embeddings, which is used for the adversarial loss
                with torch.no_grad():
                    clip_text_token = open_clip_tokenizer(text).to(accelerator.device)
                    clip_emb = open_clip_model.encode_text(clip_text_token)

                # 5. Prepare prompt embeds and unet_added_conditions
                prompt_embeds = encoded_text.pop("prompt_embeds")

                if std_item is None:
                    noisy_model_input = add_noise(latents, noise_scheduler, start_timesteps, topk, args.num_ddim_timesteps, bsz)

                # 6. Get online LCM prediction on z_{t_{n + k}} (noisy_model_input), w, c, t_{n + k} (start_timesteps)
                noise_pred = unet(
                    noisy_model_input,
                    start_timesteps,
                    timestep_cond=None,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=encoded_text,
                ).sample

                pred_x_0_stu = get_predicted_original_sample(
                    noise_pred,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )

                # 7. Set target step mode, `random_s` means using random target step; 
                # `pcm` means select target step by multi-phase; 
                # and `lcm` means select 0 as target step.
                if args.solver_mode == 'random_s':
                    random_range = math.ceil(args.num_ddim_timesteps * args.strength * args.random_range_rate)
                    model_pred, end_timesteps_index_s = solver.ddim_random_s(
                        pred_x_0_stu, noise_pred, index, random_range#args.multiphase, args.strength
                    )
                    end_timesteps = end_timesteps_index_s * topk
                    # print("==================== Selected Timestep Index ====================")
                    # print(f"index & selected random timestep index: {int(index)}, {int(end_timesteps_index_s)}")
                    # print("==================== Selected Timestep Index ====================")
                elif args.solver_mode == 'pcm':
                    model_pred, end_timesteps = solver.ddim_style_multiphase(
                        pred_x_0_stu, noise_pred, index, args.multiphase, args.strength
                    )
                elif args.solver_mode == 'lcm':
                    model_pred = pred_x_0_stu
                else:
                    raise ValueError(f'solver mode should be one of `lcm`, `pcm`, and `random_s`')
                # model_pred = c_skip_start * noisy_model_input + c_out_start * model_pred

                # 8. Compute the conditional and unconditional teacher model predictions to get CFG estimates of the
                # predicted noise eps_0 and predicted original sample x_0, then run the ODE solver using these
                # estimates to predict the data point in the augmented PF-ODE trajectory corresponding to the next ODE
                # solver timestep.
                with torch.no_grad():
                    with torch.autocast("cuda"):
                        # 1. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and conditional embedding c
                        cond_teacher_output = teacher_unet(
                            noisy_model_input.to(weight_dtype),
                            start_timesteps,
                            encoder_hidden_states=prompt_embeds.to(weight_dtype),
                            added_cond_kwargs=encoded_text,
                        ).sample
                        cond_pred_x0 = get_predicted_original_sample(
                            cond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                        cond_pred_noise = get_predicted_noise(
                            cond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )

                        # 2. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and unconditional embedding 0
                        if args.using_cfg:
                            uncond_teacher_output = teacher_unet(
                                noisy_model_input.to(weight_dtype),
                                start_timesteps,
                                encoder_hidden_states=uncond_prompt_embeds.to(weight_dtype),
                                added_cond_kwargs=encoded_text,
                            ).sample
                            uncond_pred_x0 = get_predicted_original_sample(
                                uncond_teacher_output,
                                start_timesteps,
                                noisy_model_input,
                                noise_scheduler.config.prediction_type,
                                alpha_schedule,
                                sigma_schedule,
                            )
                            uncond_pred_noise = get_predicted_noise(
                                uncond_teacher_output,
                                start_timesteps,
                                noisy_model_input,
                                noise_scheduler.config.prediction_type,
                                alpha_schedule,
                                sigma_schedule,
                            )

                        # 3. Calculate the CFG estimate of x_0 (pred_x0) and eps_0 (pred_noise)
                        # Note that this uses the LCM paper's CFG formulation rather than the Imagen CFG formulation
                        # print(f"cond_pred_x0: {cond_pred_x0.shape}; uncond_pred_x0: {uncond_pred_x0.shape}; cond_pred_noise: {cond_pred_noise.shape}; uncond_pred_noise: {uncond_pred_noise.shape}; w: {w.shape}")
                            pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                            pred_noise = cond_pred_noise + w * (
                                cond_pred_noise - uncond_pred_noise
                            )
                        else:
                            pred_x0 = cond_pred_x0
                            pred_noise = cond_pred_noise
                        # 4. Run one step of the ODE solver to estimate the next point x_prev on the
                        # augmented PF-ODE trajectory (solving backward in time)
                        # Note that the DDIM step depends on both the predicted x_0 and source noise eps_0.
                        x_prev = solver.ddim_step(pred_x0, pred_noise, index)
                
                # STD: update std_bank
                if args.use_std:
                    save_encoded_text = deepcopy(encoded_text)
                    save_encoded_text.update(dict(prompt_embeds=prompt_embeds))
                    std_bank.update((index - 1).to('cpu'), x_prev.to('cpu'), {k: v.to('cpu') for k, v in save_encoded_text.items()}, text, pixel_values.to('cpu'), idx=selected_idx)
                    # std_bank.debug_print()
                
                # get model pred for discriminater
                if args.disc_input_mode == 'x0':
                    # model_pred_4disc = c_skip_x0 * noisy_model_input + c_out_x0 * pred_x_0_stu
                    model_pred_4disc = pred_x_0_stu
                elif args.disc_input_mode == 'xs':
                    if args.solver_mode == 'lcm':
                        end_timesteps = topk * torch.Tensor([np.random.randint(0, int(index+1))]).to(model_pred.device).long()
                        model_pred_4disc = add_noise(pred_x_0_stu, noise_scheduler, end_timesteps, topk, args.num_ddim_timesteps, bsz)
                    else:
                        model_pred_4disc = model_pred
                    latents = encode_and_sample(pixel_values, vae, args.num_frames, args.use_motion, weight_dtype,)
                    latents = add_noise(latents, noise_scheduler, end_timesteps, topk, args.num_ddim_timesteps, bsz)
                    pixel_values = sample_and_decode(latents, vae, args.num_frames, args.num_frames, args.use_motion, weight_dtype)
                    if args.use_motion:
                        pixel_values = rearrange(pixel_values, "(b t) c h w -> b c t h w", t=args.num_frames)
                elif args.disc_input_mode == 'xs-x0':
                    if args.solver_mode == 'lcm':
                        end_timesteps = topk * torch.Tensor([np.random.randint(0, int(index+1))]).to(model_pred.device).long()
                        model_pred_4disc = add_noise(pred_x_0_stu, noise_scheduler, end_timesteps, topk, args.num_ddim_timesteps, bsz)
                    else:
                        model_pred_4disc = model_pred
                elif args.disc_input_mode == "xs-xr":  # r < s
                    if args.solver_mode == 'lcm':
                        end_timesteps = topk * torch.Tensor([np.random.randint(0, int(index+1))]).to(model_pred.device).long()
                        model_pred_4disc = add_noise(pred_x_0_stu, noise_scheduler, end_timesteps, topk, args.num_ddim_timesteps, bsz)
                    else:
                        model_pred_4disc = model_pred
                    end_timesteps_r = (np.random.rand() * end_timesteps).long()
                    latents = encode_and_sample(pixel_values, vae, args.num_frames, args.use_motion, weight_dtype,)
                    latents = add_noise(latents, noise_scheduler, end_timesteps_r, topk, args.num_ddim_timesteps, bsz)
                    pixel_values = sample_and_decode(latents, vae, args.num_frames, args.num_frames, args.use_motion, weight_dtype)
                    if args.use_motion:
                        pixel_values = rearrange(pixel_values, "(b t) c h w -> b c t h w", t=args.num_frames)
                elif args.disc_input_mode == "xs-xR":  # r > s
                    if args.solver_mode == 'lcm':
                        end_timesteps = topk * torch.Tensor([np.random.randint(0, int(index+1))]).to(model_pred.device).long()
                        model_pred_4disc = add_noise(pred_x_0_stu, noise_scheduler, end_timesteps, topk, args.num_ddim_timesteps, bsz)
                    else:
                        model_pred_4disc = model_pred
                    end_timesteps_r = (end_timesteps + np.random.rand() * (start_timesteps - end_timesteps)).long()
                    latents = encode_and_sample(pixel_values, vae, args.num_frames, args.use_motion, weight_dtype,)
                    latents = add_noise(latents, noise_scheduler, end_timesteps_r, topk, args.num_ddim_timesteps, bsz)
                    pixel_values = sample_and_decode(latents, vae, args.num_frames, args.num_frames, args.use_motion, weight_dtype)
                    if args.use_motion:
                        pixel_values = rearrange(pixel_values, "(b t) c h w -> b c t h w", t=args.num_frames)
                    

                # 9. Get target LCM prediction on x_prev, w, c, t_n (timesteps)
                # Note that we do not use a separate target network for LCM-LoRA distillation.
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=weight_dtype):
                        target_noise_pred = unet(
                            x_prev.float(),
                            timesteps,
                            timestep_cond=None,
                            encoder_hidden_states=prompt_embeds.float(),
                            added_cond_kwargs=encoded_text,
                        ).sample
                    pred_x_0 = get_predicted_original_sample(
                        target_noise_pred,
                        timesteps,
                        x_prev,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    # target = c_skip * x_prev + c_out * pred_x_0
                    if args.solver_mode == 'random_s':
                        target, _ = solver.ddim_random_s(
                            pred_x_0, target_noise_pred, index, random_range, end_timesteps_index_s, #args.multiphase, args.strength
                        )
                    elif args.solver_mode == 'pcm':
                        target, end_timesteps = solver.ddim_style_multiphase(
                            pred_x_0, target_noise_pred, index, args.multiphase, args.strength
                        )
                    elif args.solver_mode == 'lcm':
                        target = pred_x_0
                    else:
                        raise ValueError(f'solver mode should be one of `lcm`, `pcm`, and `random_s`')
                    # target = c_skip * x_prev + c_out * target
                
                if args.use_monitor is True and (step % (args.monitor_steps * args.gradient_accumulation_steps) == 0) and accelerator.is_main_process:

                    # add monitor files (pixel_values, pred_x_0_stu, model_pred, pred_x_0, target)
                    save_name = f"{global_step:07d}.jpg"
                    if args.use_motion:
                        pixel_values_monitor = rearrange(pixel_values, "b c t h w -> (b t) c h w")
                        batch_size_monitor = args.num_frames
                    else:
                        pixel_values_monitor = pixel_values
                        batch_size_monitor = 6
                    with torch.no_grad():
                        pred_x_0_stu_monitor = sample_and_decode(pred_x_0_stu, vae, args.num_frames, args.num_frames, args.use_motion, weight_dtype)
                        model_pred_monitor = sample_and_decode(model_pred, vae, args.num_frames, args.num_frames, args.use_motion, weight_dtype)
                        pred_x_0_monitor = sample_and_decode(pred_x_0, vae, args.num_frames, args.num_frames, args.use_motion, weight_dtype)
                        target_monitor = sample_and_decode(target, vae, args.num_frames, args.num_frames, args.use_motion, weight_dtype)
                        model_pred_4disc_monitor = sample_and_decode(model_pred_4disc, vae, args.num_frames, args.num_frames, args.use_motion, weight_dtype)

                    train_monitor.update(key="batch_tensor", 
                                        data=torch.cat([
                                            pixel_values_monitor, 
                                            pred_x_0_stu_monitor, 
                                            model_pred_monitor,
                                            pred_x_0_monitor,
                                            target_monitor,
                                            model_pred_4disc_monitor,
                                        ], dim=0),
                                        save_name=save_name, 
                                        batch_size=batch_size_monitor)
                    train_monitor.update(
                        key='text',
                        data=dict(
                            global_step=global_step,
                            text=text,
                        ),
                        save_name='text',
                    )
                    print(f"Update {save_name} to monitor, current xs-timestep is {end_timesteps}")

                # 10. Calculate the CD loss and discriminator loss
                loss_dict = {}

                # 10.1. Calculate CD loss
                if args.cd_target in ["learn", "hlearn"] and args.use_motion:
                    model_pred_cd = spatial_head(model_pred.float())
                else:
                    model_pred_cd = model_pred.float()
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=weight_dtype):
                        if args.cd_target in ["learn", "hlearn"] and args.use_motion:
                            target_cd = target_spatial_head(target.float())
                        else:
                            target_cd = target.float()
                if args.loss_type == "l2":
                    loss_unet_cd = F.mse_loss(
                        model_pred_cd.float(), target_cd.float(), reduction="mean"
                    )
                elif args.loss_type == "huber":
                    loss_unet_cd = torch.mean(
                        torch.sqrt(
                            (model_pred_cd.float() - target_cd.float()) ** 2
                            + args.huber_c**2
                        )
                        - args.huber_c
                    )
                loss_dict["loss_unet_cd"] = loss_unet_cd
                loss_unet_total = loss_unet_cd

                # 10.2. Calculate discriminator loss
                if global_step >= args.disc_start_step and not args.no_disc:
                    model_pred_pixel = sample_and_decode(
                        model_pred_4disc,
                        vae,
                        args.num_frames,
                        args.disc_tsn_num_frames,
                        args.use_motion, 
                        weight_dtype,
                    )

                    clip_emb = repeat(
                        clip_emb, "b n -> b t n", t=args.disc_tsn_num_frames
                    )
                    clip_emb = rearrange(clip_emb, "b t n -> (b t) n")

                    gen_dino_features, gen_sample = get_dino_features(
                        model_pred_pixel,
                        normalize_fn=normalize_fn,
                        dino_model=dino,
                        dino_hooks=args.disc_dino_hooks,
                        return_cls_token=False,
                    )
                    disc_pred_gen = discriminator(
                        gen_dino_features, clip_emb, return_key_list=["logits"]
                    )["logits"]

                    if args.disc_loss_type == "bce":
                        pos_label = torch.ones_like(disc_pred_gen)
                        loss_unet_adv = F.binary_cross_entropy_with_logits(
                            disc_pred_gen, pos_label
                        )
                    elif args.disc_loss_type == "hinge":
                        loss_unet_adv = -disc_pred_gen.mean() + 1
                    elif args.disc_loss_type == "wgan":
                        loss_unet_adv = -disc_pred_gen.mean()
                    else:
                        raise ValueError(
                            f"Discriminator loss type {args.disc_loss_type} not supported."
                        )

                    loss_dict["loss_unet_adv"] = loss_unet_adv
                    loss_unet_total = (
                        loss_unet_total + args.disc_loss_weight * loss_unet_adv
                    )

                loss_dict["loss_unet_total"] = loss_unet_total

                # 11. Backpropagate on the online student model (`unet`)
                accelerator.backward(loss_unet_total)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # 12. Train the discriminator
                if global_step >= args.disc_start_step and not args.no_disc:
                    disc_optimizer.zero_grad(set_to_none=True)

                    with torch.no_grad():
                        gen_sample = sample_and_decode(
                            model_pred_4disc.detach(),
                            vae,
                            args.num_frames,
                            args.disc_tsn_num_frames,
                            args.use_motion,
                            weight_dtype,
                        )

                    # get GT samples
                    if args.disc_gt_data == "webvid":
                        if args.use_motion:
                            tsn_sample_indices = tsn_sample(
                                args.num_frames, args.disc_tsn_num_frames
                            )
                            pixel_values = pixel_values[:, :, tsn_sample_indices]
                            gt_sample = rearrange(pixel_values, "b c t h w -> (b t) c h w")
                        else:
                            gt_sample = pixel_values
                        gt_sample_clip_emb = clip_emb
                    else:
                        gt_sample = gt_sample.to(
                            accelerator.device, dtype=weight_dtype, non_blocking=True
                        )
                        with torch.no_grad():
                            gt_sample_clip_text_token = open_clip_tokenizer(
                                gt_sample_caption
                            ).to(accelerator.device)
                            gt_sample_clip_emb = open_clip_model.encode_text(
                                gt_sample_clip_text_token
                            )
                    if args.disc_gt == 'target': # gt or target
                        gt_sample = sample_and_decode(target, vae, args.num_frames, args.num_frames, args.use_motion, weight_dtype)

                    # get discriminator predictions on generated sampels
                    with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype):
                        gen_dino_features, gen_sample = get_dino_features(
                            gen_sample,
                            normalize_fn=normalize_fn,
                            dino_model=dino,
                            dino_hooks=args.disc_dino_hooks,
                            return_cls_token=False,
                        )
                    disc_pred_gen = discriminator(
                        gen_dino_features, clip_emb, return_key_list=["logits"]
                    )["logits"]

                    # get discriminator predictions on GT samples
                    with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype):
                        gt_dino_features, processed_gt_sample = get_dino_features(
                            gt_sample,
                            normalize_fn=normalize_fn,
                            dino_model=dino,
                            dino_hooks=args.disc_dino_hooks,
                            return_cls_token=False,
                        )
                    disc_pred_gt = discriminator(
                        gt_dino_features, gt_sample_clip_emb, return_key_list=["logits"]
                    )["logits"]

                    if args.disc_loss_type == "bce":
                        pos_label = torch.ones_like(disc_pred_gen)
                        neg_label = torch.zeros_like(disc_pred_gen)
                        loss_disc_gt = F.binary_cross_entropy_with_logits(
                            disc_pred_gt, pos_label
                        )
                        loss_disc_gen = F.binary_cross_entropy_with_logits(
                            disc_pred_gen, neg_label
                        )
                    elif args.disc_loss_type == "hinge":
                        loss_disc_gt = (
                            torch.max(torch.zeros_like(disc_pred_gt), 1 - disc_pred_gt)
                        ).mean()
                        loss_disc_gen = (
                            torch.max(torch.zeros_like(disc_pred_gt), 1 + disc_pred_gen)
                        ).mean()
                    elif args.disc_loss_type == "wgan":
                        loss_disc_gt = (
                            torch.max(-torch.ones_like(disc_pred_gt), -disc_pred_gt)
                        ).mean()
                        loss_disc_gen = (
                            torch.max(-torch.ones_like(disc_pred_gt), disc_pred_gen)
                        ).mean()
                    else:
                        raise ValueError(
                            f"Discriminator loss type {args.disc_loss_type} not supported."
                        )

                    loss_disc_total = loss_disc_gt + loss_disc_gen
                    loss_dict["loss_disc_gt"] = loss_disc_gt
                    loss_dict["loss_disc_gen"] = loss_disc_gen

                    if args.disc_lambda_r1 > 0:
                        # not sure if this is the correct way to calculate the gradient penalty
                        with torch.autocast("cuda", dtype=weight_dtype):
                            alpha = torch.rand(
                                int(bsz * args.disc_tsn_num_frames),
                                1,
                                1,
                                1,
                                device=gt_sample.device,
                            )
                            interpolations = (
                                alpha * processed_gt_sample + (1 - alpha) * gen_sample
                            )
                            interpolation_features, interpolations = get_dino_features(
                                interpolations,
                                normalize_fn=normalize_fn,
                                dino_model=dino,
                                dino_hooks=args.disc_dino_hooks,
                                return_cls_token=False,
                                preprocess_sample=False,
                            )
                            for feat_idx in range(len(interpolation_features)):
                                interpolation_features[feat_idx].requires_grad = True

                            disc_interpolation_logit_list = discriminator(
                                interpolation_features,
                                clip_emb,
                                return_key_list=["logit_list"],
                            )["logit_list"]

                            gradients = torch.autograd.grad(
                                outputs=[
                                    item.sum() for item in disc_interpolation_logit_list
                                ],
                                inputs=interpolation_features,
                                create_graph=True,
                            )
                            gradients = torch.cat(gradients, dim=1)
                            gradients = gradients.reshape(gradients.size(0), -1)
                            grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                        # adjust disc_lambda_r1
                        if global_step - last_update_r1_step > 500:
                            if grad_penalty >= 100:
                                args.disc_lambda_r1 = args.disc_lambda_r1 * 5.0
                                last_update_r1_step = global_step
                                logger.warning(
                                    f"Graident penalty too high, increasing disc_lambda_r1 to {args.disc_lambda_r1}"
                                )

                        loss_dict["loss_disc_r1"] = grad_penalty
                        loss_disc_total = (
                            loss_disc_total + args.disc_lambda_r1 * grad_penalty
                        )

                    loss_dict["loss_disc_total"] = loss_disc_total

                    accelerator.backward(loss_disc_total)
                    disc_optimizer.step()
                    disc_lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # 13. Make EMA update to target student model parameters (`target_unet`)
                if args.cd_target in ["learn", "hlearn"] and args.use_motion:
                    update_ema(
                        target_spatial_head.parameters(),
                        spatial_head.parameters(),
                        args.ema_decay,
                    )
                progress_bar.update(1)
                global_step += 1

                # according to https://github.com/huggingface/diffusers/issues/2606
                # DeepSpeed need to run save for all processes
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d
                                for d in checkpoints
                                if (
                                    d.startswith("checkpoint")
                                    and "step" not in d
                                    and "final" not in d
                                )
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                    accelerator.wait_for_everyone()
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    try:
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    except Exception as e:
                        logger.info(f"Failed to save state: {e}")

            logs = {
                "unet_lr": lr_scheduler.get_last_lr()[0],
                "disc_lr": disc_lr_scheduler.get_last_lr()[0],
                "disc_r1_weight": args.disc_lambda_r1,
            }
            for loss_name, loss_value in loss_dict.items():
                if type(loss_value) == torch.Tensor:
                    logs[loss_name] = loss_value.item()
                else:
                    logs[loss_name] = loss_value

            current_time = datetime.now().strftime("%m-%d-%H:%M")
            progress_bar.set_postfix(
                **logs,
                **{"cur time": current_time},
                **{"video_name": batch["__key__"]},
            )
            try:
                accelerator.log(logs, step=global_step)
            except Exception as e:
                logger.info(f"Failed to log metrics at step {global_step}: {e}")

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(os.path.join(args.output_dir, "checkpoint-final"))
        lora_state_dict = get_peft_model_state_dict(unet, adapter_name="default")
        StableDiffusionPipeline.save_lora_weights(
            os.path.join(args.output_dir, "checkpoint-final", "unet_lora"),
            lora_state_dict,
        )
        if args.cd_target in ["learn", "hlearn"] and args.use_motion:
            spatial_head_ = accelerator.unwrap_model(spatial_head)
            spatial_head_.save_pretrained(
                os.path.join(args.output_dir, "spatial_head")
            )
            target_spatial_head_ = accelerator.unwrap_model(target_spatial_head)
            target_spatial_head_.save_pretrained(
                os.path.join(args.output_dir, "target_spatial_head")
            )

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    save_args_to_yaml(args, os.path.join(args.output_dir, 'config.yaml'))
    main(args)
