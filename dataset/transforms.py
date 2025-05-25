import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
import random

class HandwritingAugmentation:
    def __init__(self, config):
        self.rotation = config.get('rotation', 0)
        self.shear = config.get('shear', 0)
        self.noise = config.get('noise', 0)
        self.brightness = config.get('brightness', 0)
        self.enabled = config.get('enabled', True)
        
    def __call__(self, image):
        if not self.enabled or random.random() > 0.7:
            return image
            
        # Convert to PIL if tensor
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image.squeeze())
        
        # Rotation augmentation
        if self.rotation > 0:
            angle = random.uniform(-self.rotation, self.rotation)
            image = image.rotate(angle, fillcolor=255)
        
        # Add slight noise
        if self.noise > 0:
            image = transforms.ToTensor()(image)
            noise = torch.randn_like(image) * self.noise
            image = torch.clamp(image + noise, 0, 1)
            image = transforms.ToPILImage()(image)
        
        # Convert back to tensor
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
            
        return image