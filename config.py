"""
Simple Configuration for MMNet Pipeline
"""

import os
import torch

# Dataset paths
DATA_BASE_PATH = './data/breakhis'
SLIDES_PATH = 'BreaKHis_v1/BreaKHis_v1/histology_slides/breast'

# Image settings
IMAGE_SIZE = 224
MAGNIFICATIONS = ['40X', '100X', '200X', '400X']

# Training settings
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

# Model settings
BACKBONE = 'resnet50'
FEATURE_DIM = 512
NUM_BINARY_CLASSES = 2
NUM_SUBTYPE_CLASSES = 9

# Output paths
OUTPUT_DIR = './output'

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

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
        'pin_memory': True if device.type == 'cuda' else False
    }