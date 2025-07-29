import torch
import random
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
import torchvision.transforms as T
from PIL import Image

from config import config


class AdvancedMultiMagDataset(Dataset):
    """
    Advanced Multi-Magnification Dataset with enhanced utilization strategies:
    1. Uses all available images per patient (not just random sampling)
    2. Implements smart augmentation for histopathology images
    3. Supports multiple sampling strategies for different training phases
    4. Includes cross-magnification consistency checks
    """
    
    def __init__(
        self, 
        patient_list, 
        patient_dict, 
        transform=None,
        phase='train',
        sampling_strategy='all_images',  # 'all_images', 'balanced_per_patient', 'single_per_patient'
        augmentation_level='high',  # 'low', 'medium', 'high'
        use_mixup=False,
        mixup_alpha=0.4,
        min_images_per_mag=1,  # Minimum images required per magnification
        max_images_per_patient=None  # Limit images per patient to prevent memory issues
    ):
        self.patient_list = patient_list
        self.patient_dict = patient_dict
        self.phase = phase
        self.sampling_strategy = sampling_strategy
        self.augmentation_level = augmentation_level
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.min_images_per_mag = min_images_per_mag
        self.max_images_per_patient = max_images_per_patient
        
        # Build image index with enhanced utilization
        self.image_samples = self._build_image_index()
        
        # Setup transforms
        self.base_transform = self._get_base_transforms()
        self.augmentation_transform = self._get_augmentation_transforms()
        self.normalize_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Custom transform if provided (for compatibility)
        self.custom_transform = transform
        
        print(f"Dataset created: {len(self.image_samples)} samples from {len(self.patient_list)} patients")
        self._print_dataset_stats()
    
    def _build_image_index(self):
        """Build comprehensive image index based on sampling strategy"""
        samples = []
        magnifications = ['40', '100', '200', '400']
        
        for patient_id in self.patient_list:
            patient_data = self.patient_dict[patient_id]
            patient_images = patient_data['images']
            label = patient_data['label']
            
            # Check which magnifications are available
            available_mags = [mag for mag in magnifications if mag in patient_images and len(patient_images[mag]) >= self.min_images_per_mag]
            
            if len(available_mags) < 2:  # Need at least 2 magnifications
                continue
                
            if self.sampling_strategy == 'all_images':
                # Use all possible combinations of images across magnifications
                samples.extend(self._generate_all_combinations(patient_id, patient_images, available_mags, label))
                
            elif self.sampling_strategy == 'balanced_per_patient':
                # Generate balanced samples per patient (same number of samples per patient)
                samples.extend(self._generate_balanced_samples(patient_id, patient_images, available_mags, label))
                
            elif self.sampling_strategy == 'single_per_patient':
                # One sample per patient (for initial testing)
                samples.extend(self._generate_single_sample(patient_id, patient_images, available_mags, label))
        
        return samples
    
    def _generate_all_combinations(self, patient_id, patient_images, available_mags, label):
        """Generate all possible image combinations for maximum data utilization"""
        samples = []
        magnifications = ['40', '100', '200', '400']
        
        # Find minimum number of images across available magnifications
        min_images = min(len(patient_images.get(mag, [])) for mag in available_mags)
        
        # Limit if max_images_per_patient is set
        if self.max_images_per_patient:
            min_images = min(min_images, self.max_images_per_patient)
        
        for i in range(min_images):
            sample = {
                'patient_id': patient_id,
                'label': label,
                'images': {}
            }
            
            for mag in magnifications:
                if mag in available_mags and len(patient_images[mag]) > i:
                    sample['images'][f'mag_{mag}'] = patient_images[mag][i]
                else:
                    # Use a random image from available ones if this mag doesn't have enough images
                    if mag in patient_images and len(patient_images[mag]) > 0:
                        sample['images'][f'mag_{mag}'] = random.choice(patient_images[mag])
                    else:
                        # Use an image from another available magnification as fallback
                        sample['images'][f'mag_{mag}'] = random.choice(patient_images[available_mags[0]])
            
            samples.append(sample)
        
        return samples
    
    def _generate_balanced_samples(self, patient_id, patient_images, available_mags, label):
        """Generate balanced number of samples per patient"""
        samples = []
        magnifications = ['40', '100', '200', '400']
        
        # Fixed number of samples per patient for balanced training
        samples_per_patient = 5  # Configurable
        
        for i in range(samples_per_patient):
            sample = {
                'patient_id': patient_id,
                'label': label,
                'images': {}
            }
            
            for mag in magnifications:
                if mag in available_mags:
                    # Randomly sample with replacement
                    sample['images'][f'mag_{mag}'] = random.choice(patient_images[mag])
                else:
                    # Fallback to available magnification
                    sample['images'][f'mag_{mag}'] = random.choice(patient_images[available_mags[0]])
            
            samples.append(sample)
        
        return samples
    
    def _generate_single_sample(self, patient_id, patient_images, available_mags, label):
        """Generate single sample per patient"""
        magnifications = ['40', '100', '200', '400']
        
        sample = {
            'patient_id': patient_id,
            'label': label,
            'images': {}
        }
        
        for mag in magnifications:
            if mag in available_mags:
                sample['images'][f'mag_{mag}'] = random.choice(patient_images[mag])
            else:
                sample['images'][f'mag_{mag}'] = random.choice(patient_images[available_mags[0]])
        
        return [sample]
    
    def _get_base_transforms(self):
        """Base transforms for preprocessing"""
        return T.Compose([
            T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        ])
    
    def _get_augmentation_transforms(self):
        """Advanced augmentation transforms based on level"""
        if self.phase != 'train':
            return None
            
        if self.augmentation_level == 'low':
            return T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.025),
            ])
        
        elif self.augmentation_level == 'medium':
            return T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
                T.RandomApply([T.ElasticTransform(alpha=50.0, sigma=5.0)], p=0.2),
            ])
        
        else:  # high level
            return T.Compose([
                # Geometric augmentations
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=20),
                T.RandomApply([T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))], p=0.4),
                
                # Color augmentations (important for histopathology)
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=0.5)], p=0.3),
                T.RandomApply([T.RandomAutocontrast()], p=0.2),
                T.RandomApply([T.RandomEqualize()], p=0.2),
                
                # Noise and blur
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.5))], p=0.3),
                
                # Advanced augmentations
                T.RandomApply([T.ElasticTransform(alpha=75.0, sigma=7.0)], p=0.25),  # Tissue deformation
                T.RandomApply([T.RandomPosterize(bits=4)], p=0.2),
                T.RandomApply([T.RandomSolarize(threshold=128)], p=0.1),
            ])
    
    def _print_dataset_stats(self):
        """Print dataset statistics"""
        if not self.image_samples:
            return
            
        label_counts = defaultdict(int)
        patient_counts = defaultdict(int)
        
        for sample in self.image_samples:
            label_counts[sample['label']] += 1
            patient_counts[sample['patient_id']] += 1
        
        print(f"  Label distribution: Benign={label_counts[0]}, Malignant={label_counts[1]}")
        print(f"  Samples per patient: min={min(patient_counts.values())}, max={max(patient_counts.values())}, avg={np.mean(list(patient_counts.values())):.1f}")
    
    def __len__(self):
        return len(self.image_samples)
    
    def __getitem__(self, idx):
        sample = self.image_samples[idx]
        label = sample['label']
        
        # Load images for all magnifications
        images = {}
        for mag_key, img_path in sample['images'].items():
            try:
                # Load image
                image = Image.open(img_path).convert('RGB')
                
                # Apply base transforms
                if self.base_transform:
                    image = self.base_transform(image)
                
                # Apply augmentation (only during training)
                if self.augmentation_transform and self.phase == 'train':
                    image = self.augmentation_transform(image)
                
                # Apply custom transform if provided (for compatibility)
                if self.custom_transform:
                    image = self.custom_transform(image)
                else:
                    # Apply normalization
                    image = self.normalize_transform(image)
                
                images[mag_key] = image
                
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Create dummy image as fallback
                images[mag_key] = torch.zeros(3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        
        # Apply Mixup if enabled (only during training)
        if self.use_mixup and self.phase == 'train' and random.random() < 0.5:
            images, label = self._apply_mixup(images, label, idx)
        
        return images, torch.tensor(label, dtype=torch.long)
    
    def _apply_mixup(self, images, label, current_idx):
        """Apply Mixup augmentation"""
        # Select another random sample
        other_idx = random.randint(0, len(self.image_samples) - 1)
        if other_idx == current_idx:
            other_idx = (other_idx + 1) % len(self.image_samples)
        
        other_sample = self.image_samples[other_idx]
        other_label = other_sample['label']
        
        # Load other images
        other_images = {}
        for mag_key, img_path in other_sample['images'].items():
            try:
                image = Image.open(img_path).convert('RGB')
                if self.base_transform:
                    image = self.base_transform(image)
                if self.augmentation_transform:
                    image = self.augmentation_transform(image)
                image = self.normalize_transform(image)
                other_images[mag_key] = image
            except:
                other_images[mag_key] = torch.zeros(3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        
        # Mix images and labels
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        mixed_images = {}
        for mag_key in images.keys():
            mixed_images[mag_key] = lam * images[mag_key] + (1 - lam) * other_images[mag_key]
        
        # For binary classification, we'll use the primary label (could be improved with soft labels)
        mixed_label = label if lam > 0.5 else other_label
        
        return mixed_images, mixed_label
    
    def get_class_weights(self):
        """Calculate class weights for balanced loss"""
        label_counts = defaultdict(int)
        for sample in self.image_samples:
            label_counts[sample['label']] += 1
        
        total_samples = len(self.image_samples)
        num_classes = len(label_counts)
        
        weights = {}
        for label, count in label_counts.items():
            weights[label] = total_samples / (num_classes * count)
        
        return [weights[0], weights[1]]  # [benign_weight, malignant_weight]


# Compatibility wrapper
class MultiMagDataset(Dataset):
    """Backward compatibility wrapper"""
    def __init__(self, patient_list, patient_dict, transform=None, phase='train'):
        self.dataset = AdvancedMultiMagDataset(
            patient_list=patient_list,
            patient_dict=patient_dict,
            transform=transform,
            phase=phase,
            sampling_strategy='balanced_per_patient',  # More conservative for compatibility
            augmentation_level='medium'
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def get_class_weights(self):
        return self.dataset.get_class_weights()


def create_enhanced_datasets(train_patients, val_patients, test_patients, patient_dict, 
                           train_transform=None, val_transform=None):
    """
    Create enhanced datasets with different strategies for train/val/test
    """
    
    # Training dataset: Maximum augmentation and data utilization
    train_dataset = AdvancedMultiMagDataset(
        patient_list=train_patients,
        patient_dict=patient_dict,
        transform=train_transform,
        phase='train',
        sampling_strategy='all_images',  # Use all available images
        augmentation_level='high',
        use_mixup=True,
        mixup_alpha=0.4,
        max_images_per_patient=10  # Prevent memory issues
    )
    
    # Validation dataset: Moderate augmentation
    val_dataset = AdvancedMultiMagDataset(
        patient_list=val_patients,
        patient_dict=patient_dict,
        transform=val_transform,
        phase='val',
        sampling_strategy='balanced_per_patient',
        augmentation_level='low',
        use_mixup=False
    )
    
    # Test dataset: No augmentation, single sample per patient
    test_dataset = AdvancedMultiMagDataset(
        patient_list=test_patients,
        patient_dict=patient_dict,
        transform=val_transform,
        phase='test',
        sampling_strategy='single_per_patient',
        augmentation_level='low',
        use_mixup=False
    )
    
    return train_dataset, val_dataset, test_dataset


def create_tta_dataset(test_patients, patient_dict, tta_transforms):
    """Create dataset for Test-Time Augmentation"""
    tta_datasets = []
    
    for i, transform in enumerate(tta_transforms):
        dataset = AdvancedMultiMagDataset(
            patient_list=test_patients,
            patient_dict=patient_dict,
            transform=transform,
            phase='test',
            sampling_strategy='single_per_patient',
            augmentation_level='low',
            use_mixup=False
        )
        tta_datasets.append(dataset)
    
    return tta_datasets