import os, io, csv, math, random
import numpy as np
from einops import rearrange
from decord import VideoReader
import json
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import traceback



class OpenSoraPlan(Dataset):
    def __init__(
            self,
            ann_path: str='/mnt/nj-public02/dataset/luoc/opensoraplan_v1.0/llava_path_cap_64x512x512.json', 
            video_folder: str='/mnt/nj-public02/dataset/luoc/opensoraplan_v1.0_dataset',
            style_image_folder: str='/mnt/nj-public02/dataset/iphone-v2',
            sample_size=256, 
            width=256, 
            height=256,
            sample_stride=4, 
            sample_n_frames=16,
            is_image=False, 
            process_fn=None,
        ):
        self.ann_type = ann_path.split('.')[-1] # csv or json
        assert self.ann_type in ['csv', 'json'], "only support csv or json annotation file"
        with open(ann_path, 'r') as annfile:
            if self.ann_type == 'csv':
                self.dataset = list(csv.DictReader(annfile))
            elif self.ann_type == 'json':
                self.dataset = json.load(annfile)

        self.length = len(self.dataset)
        with open(f'{style_image_folder}/style30k/list.txt', 'r') as style_file:
            lines = style_file.readlines()
            self.style_dataset = [line.strip() for line in lines]
            

        self.video_folder    = video_folder
        self.style_image_folder = style_image_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.process_fn = process_fn
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        if self.ann_type == 'csv':
            videoid = video_dict['videoid']
            video_path = os.path.join(self.video_folder, f"{videoid}.mp4")
            name = video_dict.get('name', None) or video_dict.get('caption', None)
            assert name is not None, f"no name or caption for video {videoid} in the dataset"
        elif self.ann_type == 'json': # for opensoraplan dataset
            video_path = video_dict['path'].replace("/remote-home1/dataset/data_split_tt/","")
            video_path = os.path.join(self.video_folder, video_path)
            name = video_dict['cap'][0]

            # keep only first sentence, the whole caption is too long
            name = name.split(".")[0]

        video_reader = VideoReader(video_path)
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]
        
        # read style image randomly
        style_image = Image.open(f"{self.style_image_folder}/{random.choice(self.style_dataset)}")

        return pixel_values, name, os.path.basename(video_path), style_image

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name, videoid, style_image = self.get_batch(idx) # pixel_values: (f, c, h, w)
                break

            except Exception as e:
                print(f"An error occurred: {e}")
                idx = random.randint(0, self.length-1)

        # pixel_values = self.pixel_transforms(pixel_values)
        pixel_values, original_size, crop_top_left, target_size = self.process_fn(pixel_values)
        if self.is_image is False:
            pixel_values = rearrange(pixel_values, 'f c h w -> c f h w')
        sample = dict(video=pixel_values, text=name, __key__=videoid, original_size=original_size, crop_top_left=crop_top_left, target_size=target_size, style_image=style_image)
        return sample

class LivephotoDataset(OpenSoraPlan):

    def __init__(self,
        ann_path: str='/mnt/nj-public02/dataset/iphone-v2/filtered_dataset/live_8w_json', 
        style_image_folder: str='/mnt/nj-public02/dataset/iphone-v2',
        sample_size=256, 
        width=256, 
        height=256,
        sample_stride=4, 
        sample_n_frames=16,
        is_image=False, 
        process_fn=None,
    ):
        self.dataset = []
        path_list = [os.path.join(ann_path, json_name) for json_name in os.listdir(ann_path)]
        for path in path_list:
            with open(path,"r") as f:
                for line in f:
                    sample = json.loads(line)
                    video_path = sample["source"]
                    control_path = ""
                    self.dataset.append(dict(video_path=video_path, text=sample["caption"]))

        with open(f'{style_image_folder}/style30k/list.txt', 'r') as style_file:
            lines = style_file.readlines()
            self.style_dataset = [line.strip() for line in lines]

        self.length = len(self.dataset)
        self.style_image_folder = style_image_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        
        self.process_fn = process_fn

    def get_batch(self, idx):
        video_dict = self.dataset[idx]

        video_reader = VideoReader(video_path['video_path'])
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]
        
        # read style image randomly
        style_image = Image.open(f"{self.style_image_folder}/{random.choice(self.style_dataset)}")

        return pixel_values, video_path['text'], os.path.basename(video_path), style_image

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name, videoid, style_image = self.get_batch(idx) # pixel_values: (f, c, h, w)
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)

        # pixel_values = self.pixel_transforms(pixel_values)
        pixel_values, original_size, crop_top_left, target_size = self.process_fn(pixel_values)
        if self.is_image is False:
            pixel_values = rearrange(pixel_values, 'f c h w -> c f h w')
        sample = dict(video=pixel_values, text=name, __key__=videoid, original_size=original_size, crop_top_left=crop_top_left, target_size=target_size, style_image=style_image)
        return sample



if __name__ == "__main__":

    dataset = OpenSoraPlan(
        ann_path='/mnt/nj-public02/dataset/luoc/opensoraplan_v1.0/llava_path_cap_64x512x512.json', 
        video_folder='/mnt/nj-public02/dataset/luoc/opensoraplan_v1.0_dataset',
        sample_size=512,
        sample_stride=4, 
        sample_n_frames=8,
        is_image=False,
    )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0,)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))
        # for i in range(batch["pixel_values"].shape[0]):
        #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)