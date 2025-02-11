import torch
from torchvision import transforms
import random

def fit_image_to_size(img_tensor, new_width, new_height):
    new_width = int(new_width)
    new_height = int(new_height)

    # Assuming img_tensor is of shape (C, H, W)
    original_height, original_width = img_tensor.shape[-2:]
    ratio_w = float(new_width) / original_width
    ratio_h = float(new_height) / original_height

    if ratio_w > ratio_h:
        # It must be fixed by width
        resize_width = new_width
        resize_height = round(original_height * ratio_w)
    else:
        # Fixed by height
        resize_width = round(original_width * ratio_h)
        resize_height = new_height

    # Resize the image using transforms.Resize
    resize_transform = transforms.Resize((resize_height, resize_width), antialias=True)
    img_resized = resize_transform(img_tensor)

    # Calculate random cropping boundaries
    max_left = resize_width - new_width
    max_top = resize_height - new_height
    
    left = random.randint(0, max_left)
    top = random.randint(0, max_top)
    right = left + new_width
    bottom = top + new_height

    # Crop the image
    img_cropped = img_resized[..., top:bottom, left:right]

    return img_cropped, (top, left), (new_height, new_width)