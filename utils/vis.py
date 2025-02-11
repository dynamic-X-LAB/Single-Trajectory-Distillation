import torch
import queue
from typing import Any, Union, Dict
import numpy as np
from PIL import Image
import cv2
import os
import torchvision.utils as vutils
import json

class BaseRealtimeVis:

    def __init__(
            self, 
            save_dir: str, 
            save_num: int=100,
            save_keys: list=[],
        ):

        self.save_dir = save_dir
        self.save_num = save_num
        self.save_keys = save_keys
        self.keys_manager = {}
        self.save_dirs = {}
        for key in save_keys:
            self.keys_manager.update({key: queue.Queue(maxsize=self.save_num)})
            os.makedirs(os.path.join(self.save_dir, key), exist_ok=True)
            self.save_dirs.update({key: os.path.join(self.save_dir, key)})

    def push(self, key: str, data: Any, save_name: str):
        ...
    
    def pop(self):
        ...

    def update(self, key: str, data: Any, save_name: str, **kwargs):
        
        if isinstance(data, Dict):
            with open(os.path.join(self.save_dir, f'{save_name}.jsonl'), 'a') as file:
                json_line = json.dumps(data)
                file.write(json_line + '\n')
            return

        assert key in self.save_keys

        if self.keys_manager[key].qsize() < self.save_num:

            self.push(key, data, save_name, **kwargs)
        
        else:
            
            self.pop(key)
            self.push(key, data, save_name, **kwargs)


class ImageRealtimeVis(BaseRealtimeVis):

    def push(self, key: str, data: Union[Image.Image, np.ndarray, torch.Tensor], save_name: str, **kwargs):
        
        save_dir = os.path.join(self.save_dirs[key], save_name)

        if isinstance(data, np.ndarray):

            cv2.imwrite(save_dir, data)
        
        elif isinstance(data, Image.Image):

            data.save(save_dir)
        
        elif isinstance(data, torch.Tensor):

            if "batch_size" in kwargs:

                nrow = kwargs["batch_size"]

            else:

                nrow = 4
            # breakpoint()
            vutils.save_image(data, save_dir, nrow=nrow, normalize=True)
        
        self.keys_manager[key].put(save_dir)
                
    def pop(self, key):

        last_dir = self.keys_manager[key].get()
        os.remove(last_dir)