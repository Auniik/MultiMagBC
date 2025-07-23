import os
import torch
import torch.nn

from utils.env import get_base_path

DATASET_PATH = get_base_path() + "/breakhis"
SLIDES_PATH = DATASET_PATH + "/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
FOLDS_PATH = DATASET_PATH + "/Folds.csv"

# Image settings
IMAGE_SIZE = 224
MAGNIFICATIONS = ['40X', '100X', '200X', '400X']

# Training settings
NUM_EPOCHS = 2
LEARNING_RATE = 1e-4  # Increased from 5e-5 to accelerate learning
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 7  # Restored to original for stable training
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5

# Gradient accumulation for effective larger batch sizes
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 16 * 2 = 32

# Enhanced regularization settings for balanced utilization
DROPOUT_RATE = 0.75  # Increased to prevent overfitting with more data
WEIGHT_DECAY = 3e-3  # Moderate increase for better generalization
LABEL_SMOOTHING = 0.1  # Balanced smoothing to prevent overconfidence

# Mixup augmentation settings
MIXUP_ALPHA = 0.2  # Reverted from 0.4 - moderate augmentation

# Focal loss settings for balanced utilization
FOCAL_ALPHA = 0.5   # More balanced class weighting (vs 70.7% malignant)
FOCAL_GAMMA = 3.0   # Increased focus on hard examples with more data

# Model settings
BACKBONE = 'efficientnet_b0'
NUM_BINARY_CLASSES = 2
NUM_SUBTYPE_CLASSES = 8

# Output paths
OUTPUT_DIR = './output'

# Dataset utilization settings for BALANCED maximum sampling
MAX_UTILIZATION_MODE = True  # Enable maximum dataset utilization
SAMPLES_PER_PATIENT_BALANCED = 5  # Balanced samples per patient (prevents overfitting)
EPOCH_MULTIPLIER_BALANCED = 3     # 3x diverse combinations (optimal balance)
VAL_SAMPLES_PER_PATIENT_BALANCED = 2  # Balanced validation samples

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    # elif torch.backends.mps.is_available():
    #     return torch.device('mps')
    else:
        return torch.device('cpu')

class FocalLoss(torch.nn.Module):
    """Enhanced Focal Loss for addressing severe class imbalance"""
    def __init__(self, alpha=0.25, gamma=3.0, weight=None, label_smoothing=0.3):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weight for positive class (benign)
        self.gamma = gamma  # Higher gamma for better focus on hard examples
        self.weight = weight
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # Ensure inputs are properly shaped
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), -1)
        if inputs.size(-1) == 1:
            # Binary classification case
            inputs = inputs.squeeze(-1)
            targets = targets.float()
            
            # Apply label smoothing for binary case
            if self.label_smoothing > 0:
                targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            
            # Calculate BCE with logits for numerical stability
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                inputs, targets, reduction='none'
            )
            
            # Calculate probabilities
            probs = torch.sigmoid(inputs)
            pt = torch.where(targets == 1, probs, 1 - probs)
            
            # Apply alpha weighting for class imbalance
            alpha_t = torch.where(targets == 1, 
                                torch.tensor(self.alpha, device=inputs.device), 
                                torch.tensor(1 - self.alpha, device=inputs.device))
            
            # Calculate focal loss
            focal_weight = alpha_t * (1 - pt) ** self.gamma
            focal_loss = focal_weight * bce_loss
            
        else:
            # Multi-class case
            # Apply label smoothing
            if self.label_smoothing > 0:
                num_classes = inputs.size(-1)
                targets_onehot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
                targets_smooth = targets_onehot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
                ce_loss = -(targets_smooth * torch.log_softmax(inputs, dim=1)).sum(dim=1)
            else:
                ce_loss = torch.nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
            
            # Calculate probabilities and focal weight
            pt = torch.exp(-ce_loss)
            
            # Apply alpha weighting
            if self.alpha is not None and len(targets) > 0:
                # Create alpha tensor based on target class
                alpha_t = torch.where(targets == 1, 
                                    torch.tensor(self.alpha, device=inputs.device), 
                                    torch.tensor(1 - self.alpha, device=inputs.device))
                focal_weight = alpha_t * (1 - pt) ** self.gamma
            else:
                focal_weight = (1 - pt) ** self.gamma
                
            focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()

def mixup_data(x, y, alpha=0.2, device='cuda'):
    """Implement mixup augmentation for enhanced regularization"""
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

def calculate_class_weights(train_labels):
    """Calculate class weights for handling imbalanced dataset"""
    import torch
    from collections import Counter
    
    label_counts = Counter(train_labels)
    total_samples = len(train_labels)
    num_classes = len(label_counts)
    
    # Calculate inverse frequency weights
    class_weights = []
    for class_id in sorted(label_counts.keys()):
        weight = total_samples / (num_classes * label_counts[class_id])
        class_weights.append(weight)
    
    return torch.tensor(class_weights, dtype=torch.float32)

def get_training_config():
    device = get_device()
    
    if device.type == 'cuda':
        batch_size = 16
        num_workers = 8
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
        'output_dir': OUTPUT_DIR
    }