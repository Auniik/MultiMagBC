import torchvision.transforms as T
import torch
import random
import numpy as np

class HistologyAugmentation:
    """Advanced augmentation techniques specifically for histopathology images"""
    
    def __init__(self, prob=0.3):
        self.prob = prob
        
    def __call__(self, img):
        if random.random() < self.prob:
            # Stain variation simulation
            img_array = np.array(img)
            
            # H&E stain variation (slight color shifts in hematoxylin/eosin channels)
            h_shift = np.random.uniform(-0.05, 0.05)
            e_shift = np.random.uniform(-0.05, 0.05)
            
            # Apply subtle stain variations
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] + h_shift * 255, 0, 255)  # Blue channel (H)
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] + e_shift * 255, 0, 255)  # Red channel (E)
            
            return T.ToPILImage()(torch.tensor(img_array.transpose(2, 0, 1), dtype=torch.uint8))
        return img

class MixUp:
    """MixUp augmentation for better generalization"""
    
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        
    def __call__(self, images_dict, labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = labels.size(0)
        index = torch.randperm(batch_size)
        
        mixed_images = {}
        for mag, imgs in images_dict.items():
            mixed_images[mag] = lam * imgs + (1 - lam) * imgs[index, :]
            
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_images, mixed_labels, lam

def get_transforms():
    train_transform = T.Compose([
        T.Resize((224, 224)),
        
        # Geometric augmentations (more aggressive for histology)
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=30, interpolation=T.InterpolationMode.BILINEAR),  # Increased rotation
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # Added affine
        
        # Histology-specific augmentation
        HistologyAugmentation(prob=0.4),
        
        # Enhanced color augmentations
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),  # Increased intensity
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.4),  # Increased prob
        T.RandomApply([T.ElasticTransform(alpha=80.0, sigma=8.0)], p=0.3),  # Enhanced elastic transform
        
        # Additional augmentations
        T.RandomApply([T.RandomPerspective(distortion_scale=0.2, p=0.5)], p=0.3),  # Perspective
        T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.2),  # Sharpness
        
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        
        # Enhanced random erasing
        T.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),  # Increased
        T.RandomApply([T.RandomErasing(p=1.0, scale=(0.01, 0.05), ratio=(0.5, 2.0), value=0)], p=0.1)  # Additional small erasures
    ])
    eval_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    # Test-time augmentation transform (lighter augmentations)
    tta_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=5),  # Minimal rotation for TTA
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    return train_transform, eval_transform, tta_transform

def get_mixup_fn():
    """Returns MixUp function for training"""
    return MixUp(alpha=0.4)