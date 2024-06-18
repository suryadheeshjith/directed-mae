import numpy as np 
import torch
import torchvision

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def unnormalize_image(image):
    image = torch.einsum('chw->hwc', image)
    image = image.cpu() * imagenet_std + imagenet_mean
    return torch.einsum('hwc->chw', image)

def convert_to_pil(tensor):
    # To prevent normalization by wandb logging
    return torchvision.transforms.functional.to_pil_image(tensor)