import json
import logging
import os
import random
import re
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image

logger = logging.getLogger(__name__)


def get_sha():
    """Get git current status"""
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    message = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        sha = sha[:8]
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        message = _run(["git", "log", "--pretty=format:'%s'", sha, "-1"]).replace(
            "'", ""
        )
    except Exception:
        pass

    return {"sha": sha, "status": diff, "branch": branch, "prev_commit": message}


def escape(string):
    # Replace all non-space special characters with an empty string
    no_special_chars = re.sub(r"[^\w\s]", "", string)
    # Replace all consecutive spaces with a single space
    single_space = re.sub(r"\s+", " ", no_special_chars)
    # Replace all spaces with underscores
    return re.sub(r"\s", "_", single_space)


def filter_null_in_json(path: str):
    if isinstance(path, Path):
        path = str(path)

    if not Path(path).exists():
        logger.error(f"{path} does not exist")
        return

    if not Path(path).suffix == ".json":
        logger.error(f"{path} is not a json file")
        return

    # backup the original file
    backup_path = Path(f"{path}.bak")
    backup_exits = backup_path.exists()
    with open(path, "r") as f:
        content = json.load(f)

        if backup_exits:
            logger.warning(f"Backup file {backup_path} already exists")
        else:
            backup_path.write_text(json.dumps(content, indent=4))

        # filter out null values
        content = {k: v for k, v in content.items() if v is not None}

    if not backup_exits:
        try:
            with open(path, "w") as f:
                json.dump(content, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to write to {path}: {e}")

    logger.info(f"Filtered null values in {path}")


def tsn_sample(num_frames: int, num_sel_frames: int) -> List[int]:
    # Ensure that the input is valid
    if num_sel_frames > num_frames:
        raise ValueError("num_sel_frames must be less than or equal to num_frames")

    # Create the list of numbers from 0 to num_frames - 1
    frame_list = list(range(num_frames))

    # Calculate the size of each sublist
    sublist_size = num_frames // num_sel_frames

    # Initialize the list for selected numbers
    selected_numbers = []

    # Loop through each sublist, select a random number, and add it to the selected_numbers list
    for i in range(num_sel_frames):
        # Determine the start and end index of the current sublist
        start = i * sublist_size
        # For the last sublist, extend to the end of the frame_list
        end = start + sublist_size if i != num_sel_frames - 1 else num_frames

        # Select a random number from the sublist
        selected_number = random.choice(frame_list[start:end])
        selected_numbers.append(selected_number)

    return selected_numbers


def random_sample(num_frames: int, num_sel_frames: int) -> List[int]:
    return random.sample(range(num_frames), num_sel_frames)


def calculate_adaptive_weight(nll_loss, g_loss, last_layer):
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight


def visualize_tensor(tensor, path: str, normalized_mean=None, normalized_std=None):
    # tensor is of shape [3, h, w]
    # First, we need to transform the tensor values to the range [0, 1]
    if normalized_mean is not None and normalized_std is not None:
        # assume the input values have been normalized
        if not isinstance(normalized_mean, torch.Tensor):
            normalized_mean = torch.tensor(normalized_mean).view(3, 1, 1)
        if not isinstance(normalized_std, torch.Tensor):
            normalized_std = torch.tensor(normalized_std).view(3, 1, 1)
        tensor = tensor * normalized_std + normalized_mean
    else:
        # assumer input values are within range [-1, 1]
        tensor = (tensor + 1) / 2

    # Convert the tensor to a PIL Image
    img = TF.to_pil_image(tensor)
    # Save the image to the specified path
    img.save(path)


def sample_and_decode(latent, vae, num_frames, tsn_num_frames, use_motion, weight_dtype):
    latent = latent / vae.config.scaling_factor
    # tsn_sample_indices = tsn_sample(num_frames, tsn_num_frames)
    if use_motion is True:
        tsn_sample_indices = random_sample(num_frames, tsn_num_frames)
        sampled_latent = latent[:, :, tsn_sample_indices]
        sampled_latent = rearrange(sampled_latent, "b c t h w -> (b t) c h w")
        sampled_latent = sampled_latent.to(dtype=weight_dtype)
    else:
        sampled_latent = latent.to(dtype=weight_dtype)
    pixel = vae.decode(sampled_latent).sample
    return pixel

def encode_and_sample(pixel_values, vae, num_frames, use_motion, weight_dtype):
    vae_encode_batch_size = 32
    if use_motion:
        train_batch_size = pixel_values.shape[0]
        pixel_values = rearrange(pixel_values, "b c t h w -> (b t) c h w")
        latents = []
        for i in range(0, pixel_values.shape[0], vae_encode_batch_size):
            latents.append(
                vae.encode(
                    pixel_values[i : i + vae_encode_batch_size]
                ).latent_dist.sample()
            )
        latents = torch.cat(latents, dim=0)
        latents = rearrange(latents, "(b t) c h w -> b c t h w", b=train_batch_size, t=num_frames)
    else:
        latents = vae.encode(pixel_values).latent_dist.sample()

    latents = latents * vae.config.scaling_factor
    latents = latents.to(weight_dtype)
    return latents

def add_noise(latents, noise_scheduler, timesteps, topk, num_ddim_timesteps, bsz):
    noise = torch.randn_like(latents)
    noisy_model_input_list = []
    for b_idx in range(bsz):
        if timesteps[b_idx] // topk != num_ddim_timesteps - 1:
            noisy_model_input = noise_scheduler.add_noise(
                latents[b_idx, None],
                noise[b_idx, None],
                timesteps[b_idx, None],
            )
        else:
            # hard swap input to pure noise to ensure zero terminal SNR
            noisy_model_input = noise[b_idx, None]
        noisy_model_input_list.append(noisy_model_input)
    noisy_model_input = torch.cat(noisy_model_input_list, dim=0)
    return noisy_model_input


def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs
