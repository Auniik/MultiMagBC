import torchvision.transforms as T

def get_transforms():
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
        T.RandomApply([T.ElasticTransform(alpha=50.0, sigma=5.0)], p=0.2),  # Simulate tissue deformation
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        T.RandomErasing(p=0.1, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random')  # Occlusion
    ])
    eval_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    return train_transform, eval_transform


def create_transforms():
    """Create advanced augmentation transforms for 96%+ accuracy"""
    
    # Advanced augmentation pipeline
    train_transform = T.Compose([
        T.Resize((224, 224)),
        # Geometric augmentations
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=20),  # More aggressive rotation
        T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1))], p=0.4),
        # Color augmentations
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.15),  # Stronger color jitter
        T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=0.5)], p=0.3),
        T.RandomApply([T.RandomAutocontrast()], p=0.2),
        T.RandomApply([T.RandomEqualize()], p=0.2),
        # Noise and blur
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.5))], p=0.3),
        # Advanced augmentations
        T.RandomApply([T.RandomPosterize(bits=4)], p=0.2),
        T.RandomApply([T.RandomSolarize(threshold=128)], p=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Enhanced TTA transforms for maximum performance
    tta_transforms = [
        # Original
        T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        # Flips
        T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        T.Compose([T.Resize((224, 224)), T.RandomVerticalFlip(p=1.0), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(p=1.0), T.RandomVerticalFlip(p=1.0), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        # Rotations
        T.Compose([T.Resize((224, 224)), T.RandomRotation(degrees=10), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        T.Compose([T.Resize((224, 224)), T.RandomRotation(degrees=(-10, 10)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        # Color variations
        T.Compose([T.Resize((224, 224)), T.ColorJitter(brightness=0.15), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        T.Compose([T.Resize((224, 224)), T.ColorJitter(contrast=0.15), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        T.Compose([T.Resize((224, 224)), T.ColorJitter(brightness=0.1, contrast=0.1), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        # Geometric transforms
        T.Compose([T.Resize((224, 224)), T.RandomAffine(degrees=0, translate=(0.08, 0.08)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        T.Compose([T.Resize((224, 224)), T.RandomAffine(degrees=0, scale=(0.95, 1.05)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        # Combined transforms
        T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(p=1.0), T.ColorJitter(brightness=0.1), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        T.Compose([T.Resize((224, 224)), T.RandomRotation(degrees=5), T.ColorJitter(contrast=0.1), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    ]
    
    eval_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, eval_transform, tta_transforms