#!/usr/bin/env python3
"""
Lightweight transforms optimized for high-end GPUs (like 4090)
Reduces data loading bottlenecks while maintaining augmentation quality
"""

import torchvision.transforms as T
import torch
import random

class FastHistologyAugmentation:
    """GPU-optimized histology augmentation"""
    
    def __init__(self, prob=0.3):
        self.prob = prob
        
    def __call__(self, img):
        # Only apply if we win the probability
        if random.random() < self.prob:
            # Simple tensor-based operations (faster than numpy)
            if isinstance(img, torch.Tensor) and img.dim() == 3:
                # Lightweight stain shift
                noise = torch.randn(3, 1, 1) * 0.02
                img = torch.clamp(img + noise, 0, 1)
        return img

def get_fast_transforms():
    """Ultra-fast transforms optimized for 4090 training"""
    
    # Core transforms only - remove expensive operations
    train_transform = T.Compose([
        T.Resize((224, 224)),
        
        # Essential geometric augmentations (fast)
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5), 
        T.RandomRotation(degrees=15),  # Reduced from 30
        
        # Lightweight color jitter
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
        
        # Single efficient augmentation
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        
        # Fast post-tensor augmentation
        FastHistologyAugmentation(prob=0.3),
        
        # Single random erasing
        T.RandomErasing(p=0.1, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0)
    ])
    
    eval_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    # Minimal TTA - just flips
    tta_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    return train_transform, eval_transform, tta_transform

def get_mixup_fn():
    """Fast mixup for training"""
    from preprocess.preprocess import MixUp
    return MixUp(alpha=0.2)