import os
import torch
import torch.nn

from utils.env import get_base_path

DATASET_PATH = get_base_path() + "/breakhis"
SLIDES_PATH = DATASET_PATH + "/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"

# Image settings
IMAGE_SIZE = 224
MAGNIFICATIONS = ['40X', '100X', '200X', '400X']

# OPTIMIZED Training settings for 95% accuracy target
NUM_EPOCHS = 50  # Increased for better convergence
LEARNING_RATE = 2e-4  # Slightly higher for faster convergence
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 12  # Increased patience for better convergence
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.3  # More aggressive LR reduction

# Gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 1

# OPTIMIZED regularization settings
DROPOUT_RATE = 0.5  # Reduced from 0.75 - was too aggressive
WEIGHT_DECAY = 1e-3  # Reduced for better learning
LABEL_SMOOTHING = 0.05  # Reduced smoothing

# Mixup settings
MIXUP_ALPHA = 0.15  # Reduced for more stability

# OPTIMIZED Focal loss settings for severely imbalanced BreakHis dataset (2.5x imbalance)
FOCAL_ALPHA = 0.75  # Strong emphasis on minority class (benign)
FOCAL_GAMMA = 4.0   # High focus on hard examples due to severe imbalance

# Model settings
BACKBONE = 'efficientnet_b0'
NUM_BINARY_CLASSES = 2
NUM_SUBTYPE_CLASSES = 8

# Output paths
OUTPUT_DIR = './output'

# BALANCED dataset utilization
SAMPLES_PER_PATIENT_BALANCED = 6  # Increased for better representation
EPOCH_MULTIPLIER_BALANCED = 2
VAL_SAMPLES_PER_PATIENT_BALANCED = 3  # Increased for better validation
MAX_IMAGES_PER_PATIENT = 150  # Increased cap

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

class OptimizedFocalLoss(torch.nn.Module):
    """Optimized Focal Loss for severely imbalanced BreakHis dataset"""
    def __init__(self, alpha=0.75, gamma=4.0, weight=None, label_smoothing=0.05):
        super(OptimizedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), -1)
        if inputs.size(-1) == 1:
            inputs = inputs.squeeze(-1)
            targets = targets.float()
            
            # Minimal label smoothing
            if self.label_smoothing > 0:
                targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                inputs, targets, reduction='none'
            )
            
            probs = torch.sigmoid(inputs)
            pt = torch.where(targets == 1, probs, 1 - probs)
            
            # Optimized alpha weighting
            alpha_t = torch.where(targets == 1, 
                                torch.tensor(self.alpha, device=inputs.device), 
                                torch.tensor(1 - self.alpha, device=inputs.device))
            
            focal_weight = alpha_t * (1 - pt) ** self.gamma
            focal_loss = focal_weight * bce_loss
            
        else:
            # Multi-class case
            if self.label_smoothing > 0:
                num_classes = inputs.size(-1)
                targets_onehot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
                targets_smooth = targets_onehot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
                ce_loss = -(targets_smooth * torch.log_softmax(inputs, dim=1)).sum(dim=1)
            else:
                ce_loss = torch.nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
            
            pt = torch.exp(-ce_loss)
            
            if self.alpha is not None and len(targets) > 0:
                alpha_t = torch.where(targets == 1, 
                                    torch.tensor(self.alpha, device=inputs.device), 
                                    torch.tensor(1 - self.alpha, device=inputs.device))
                focal_weight = alpha_t * (1 - pt) ** self.gamma
            else:
                focal_weight = (1 - pt) ** self.gamma
                
            focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()

# Use the optimized focal loss
FocalLoss = OptimizedFocalLoss

def mixup_data(x, y, alpha=0.15, device='cuda'):
    """Implement mixup augmentation with reduced alpha for stability"""
    import torch
    import numpy as np
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x['mag_40'].size(0)
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = {}
    for mag_key in x.keys():
        mixed_x[mag_key] = lam * x[mag_key] + (1 - lam) * x[mag_key][index, :]
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculate mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def calculate_class_weights(train_labels, method='balanced'):
    """Calculate optimized class weights"""
    import torch
    from collections import Counter
    
    label_counts = Counter(train_labels)
    total_samples = len(train_labels)
    num_classes = len(label_counts)
    
    if method == 'balanced':
        class_weights = []
        for class_id in sorted(label_counts.keys()):
            weight = total_samples / (num_classes * label_counts[class_id])
            class_weights.append(weight)
    elif method == 'optimized':
        # Optimized weighting for your specific data
        class_weights = []
        for class_id in sorted(label_counts.keys()):
            base_weight = total_samples / (num_classes * label_counts[class_id])
            if class_id == 0:  # Benign class
                weight = base_weight * 1.3  # Moderate emphasis on benign
            else:  # Malignant class  
                weight = base_weight * 0.7  # Slight de-emphasis on malignant
            class_weights.append(weight)
    
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"📊 Class weights ({method}): Benign={weights_tensor[0]:.2f}, Malignant={weights_tensor[1]:.2f}")
    print(f"   Weight ratio: {weights_tensor[0]/weights_tensor[1]:.1f}x")
    
    return weights_tensor

def get_training_config():
    device = get_device()
    
    if device.type == 'cuda':
        batch_size = 48  # Reduced for more stable gradients
        num_workers = 12  # Reduced for stability
        environment = 'cuda'
    elif device.type == 'mps':
        batch_size = 8
        num_workers = 0
        environment = 'mps'
    else:
        batch_size = 4
        num_workers = 2
        environment = 'cpu'
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)
    
    return {
        'device': device,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'environment': environment,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'random_seed': RANDOM_SEED,
        'pin_memory': True if device.type == 'cuda' else False,
        'persistent_workers': True if device.type == 'cuda' and num_workers > 0 else False,
        'prefetch_factor': 3 if device.type == 'cuda' else None,
        'output_dir': OUTPUT_DIR
    }