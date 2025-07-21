"""
Utility Functions for MMNet
"""

import os
import random
import json
from datetime import datetime
import numpy as np
import torch
from config import OUTPUT_DIR

def seed_everything(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_experiment_directory(experiment_name=None):
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    exp_dir = os.path.join(OUTPUT_DIR, 'experiments', experiment_name)
    
    subdirs = ['models', 'plots', 'logs', 'results']
    created_dirs = {}
    
    for subdir in subdirs:
        dir_path = os.path.join(exp_dir, subdir)
        os.makedirs(dir_path, exist_ok=True)
        created_dirs[subdir] = dir_path
    
    return exp_dir, created_dirs

def save_metrics(metrics, filename=None, experiment_dir=None):
    if filename is None:
        filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    if experiment_dir is None:
        results_dir = os.path.join(OUTPUT_DIR, 'results')
    else:
        results_dir = os.path.join(experiment_dir, 'results')
    
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    
    metrics_with_info = {
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(filepath, 'w') as f:
        json.dump(metrics_with_info, f, indent=2, default=str)
    
    return filepath