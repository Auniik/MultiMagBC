"""
MMNet - Multi-Magnification Network for BreakHis Dataset
Main entry point for data analysis and training
"""

import random
import numpy as np
import torch
from config import get_training_config


def main():
    print("MMNet - Multi-Magnification Network for Breast Cancer Classification")
    print("=" * 60)

    from utils.helpers import seed_everything
    config = get_training_config()
    seed_everything(config['random_seed'])
    
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    # Dataset Analysis
    print("\nDataset Analysis:")
    from preprocess.analyze import analyze_dataset
    analyze_dataset()
    
    print("\nTo start training: python train.py")


if __name__ == "__main__":
    main()