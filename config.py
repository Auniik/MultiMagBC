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
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 7
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5

# Regularization settings
DROPOUT_RATE = 0.5
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.1

# Focal loss settings
FOCAL_ALPHA = 0.7  # Weight for positive class
FOCAL_GAMMA = 2.0  # Focusing parameter

# Model settings
BACKBONE = 'efficientnet_b0'
NUM_BINARY_CLASSES = 2
NUM_SUBTYPE_CLASSES = 8

# Output paths
OUTPUT_DIR = './output'

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

class FocalLoss(torch.nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.7, gamma=2.0, weight=None, label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
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
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * (1 - pt) ** self.gamma
        else:
            focal_weight = (1 - pt) ** self.gamma
            
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()

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