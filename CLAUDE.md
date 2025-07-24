# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based deep learning project for breast cancer classification using multi-magnification histopathology images from the BreakHis dataset. The project implements MMNet (Multi-Magnification Network) with attention mechanisms to process images at different magnifications (40X, 100X, 200X, 400X) simultaneously.

## Common Commands

### Setup and Environment
```bash
# Setup environment (for RunPod/cloud environments)
bash setup.sh

# Install dependencies
pip install -r requirements.txt
```

### Training and Evaluation
```bash
# Run full training pipeline with k-fold cross-validation
python main.py

# Run evaluation and generate visualizations
python eval.py

# Generate specific analysis
python eval.py --learning-curves
python eval.py --roc-curves
python eval.py --table
python eval.py --magnitude
```

## Code Architecture

### Core Components

1. **Model Architecture** (`backbones/our/model.py`):
   - `MMNet`: Main multi-magnification network with attention mechanisms
   - `HybridCrossMagFusion`: Cross-magnification fusion with attention
   - `MultiScaleAttentionPool`: Spatial attention pooling

2. **Data Pipeline** (`preprocess/`):
   - `MultiMagPatientDataset`: Handles multi-magnification patient data with sampling strategies
   - `PatientWiseKFoldSplitter`: Patient-wise k-fold splitting to prevent data leakage
   - `multimagset.py`: Core dataset implementation with balanced sampling

3. **Training Pipeline** (`training/`):
   - `train_mm_k_fold.py`: Training functions with mixup, threshold optimization
   - `train_single_mag.py`: Single magnification baseline training
   - `ensemble_utils.py`: Ensemble methods for multiple models

4. **Configuration** (`config.py`):
   - Centralized configuration including hyperparameters, loss functions, data paths
   - Custom `FocalLoss` implementation for class imbalance
   - Training configuration based on device capabilities

### Key Features

- **Multi-magnification Processing**: Processes 4 different magnifications simultaneously
- **Patient-wise Cross-validation**: Ensures no patient data leakage between folds
- **Attention Mechanisms**: Hierarchical magnification attention and cross-magnification fusion
- **Class Balancing**: Handles imbalanced dataset with focal loss and weighted sampling
- **Mixed Precision Training**: Uses AMP for faster training and lower memory usage
- **Threshold Optimization**: Automatically finds optimal classification threshold per fold

### Data Structure

The project expects the BreakHis dataset structure:
```
[data/workspace]/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/
├── benign/
│   └── SOB/
│       ├── adenosis/
│       ├── fibroadenoma/
│       ├── phyllodes_tumor/
│       └── tubular_adenoma/
└── malignant/
    └── SOB/
        ├── ductal_carcinoma/
        ├── lobular_carcinoma/
        ├── mucinous_carcinoma/
        └── papillary_carcinoma/
```

Each patient folder contains subfolders for different magnifications (40X, 100X, 200X, 400X).

### Output Structure

Results are saved to `output/` directory:
- `models/`: Best model checkpoints per fold
- `results/`: JSON results per fold and CSV summaries
- `plots/`: Training curves and analysis plots
- `gradcam/`: GradCAM visualizations for model interpretability

### Important Implementation Details

- **Patient-wise Splitting**: Uses patient IDs to ensure no data leakage between train/val/test
- **Dynamic Sampling**: Training uses adaptive sampling based on available images per patient
- **Magnification Masking**: Handles missing magnifications with zero tensors and attention masks
- **Threshold Optimization**: Uses precision-recall curve to find optimal threshold per fold
- **Early Stopping**: Monitors validation balanced accuracy with configurable patience

### Environment Variables and Paths

The project uses `utils/env.py` to handle different environments (local vs runpod environment). The base path for data is automatically detected.

### Key Hyperparameters (config.py)

- Image size: 224x224
- Batch size: Automatically adjusted based on device (16 for CUDA, 8 for MPS, 4 for CPU)
- Learning rate: 1e-4 with ReduceLROnPlateau scheduler
- Early stopping patience: 7 epochs
- Dropout rate: 0.75 for regularization
- Focal loss: α=0.5, γ=3.0 for class imbalance