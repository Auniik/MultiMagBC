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