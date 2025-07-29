# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-magnification histopathology image classification project focused on breast cancer detection using the BreakHis dataset. The project implements a lightweight CNN architecture (`MultiMagLightweightCNN`) that processes images at four different magnifications (40x, 100x, 200x, 400x) simultaneously for binary classification (benign vs malignant).

## Key Architecture Components

### Model Architecture (`backbones/our.py`)
- **MultiMagLightweightCNN**: Main model with ~200-500K parameters (vs 5M+ for EfficientNet-based models)
- **InvertedResidual**: MobileNetV2-style building blocks with depthwise separable convolutions
- **ChannelSpatialAttention**: Lightweight attention combining channel and spatial attention
- **CrossMagFusionLight**: Cross-magnification fusion with learnable attention weights
- Shared shallow features across magnifications with magnitude-specific deep processing

### Data Pipeline
- **MultiMagDataset** (`preprocess/multimagset.py`): Handles multi-magnification image loading
- **PatientWiseKFoldSplitter** (`preprocess/kfold_splitter.py`): Ensures patient-wise splits to prevent data leakage
- **Data Structure**: `dataset_dir/[benign|malignant]/hospital/subtype/patient_id/magnification/images`

### Training Framework (`training/train_k_fold.py`)
- K-fold cross-validation with patient-wise splitting
- Weighted loss for class imbalance handling
- AdamW optimizer with CosineAnnealingLR scheduler
- Comprehensive metrics tracking (accuracy, balanced accuracy, precision, recall, F1)

## Common Development Commands

### Environment Setup
```bash
# For RunPod environment
bash setup.sh

# Install dependencies
pip install -r requirements.txt  # or requirements.runpod for RunPod
```

### Training Commands

**Enhanced Training (Recommended for 96% accuracy):**
```bash
# Full enhanced training with all optimizations
python main_enhanced.py

# Quick test run (reduced epochs)
python main_enhanced.py --quick-test

# Single fold test
python main_enhanced.py --single-fold

# Without test-time augmentation
python main_enhanced.py --no-tta
```

**Standard Training:**
```bash
# Original training script
python main.py

# Direct training call  
python training/train_k_fold.py

# Enhanced training directly
python training/enhanced_train_k_fold.py
```

### Configuration
All hyperparameters are centralized in `config.py`:
- Model: `MODEL_NAME`, `BASE_CHANNELS`, `DROPOUT`
- Training: `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`
- Data: `MAGNIFICATIONS`, `IMAGE_SIZE`, `N_SPLITS`
- Paths: `DATASET_DIR`, `OUTPUT_DIR`, `LOGS_DIR`, `MODELS_DIR`, `RESULTS_DIR`

### Evaluation and Visualization
- **GradCAM**: `evaluate/gradcam.py` for attention visualization
- **Output Structure**: 
  - `output/models/`: Best model checkpoints per fold
  - `output/logs/`: Training logs per fold  
  - `output/results/`: Test metrics and predictions per fold
  - `output/plots/`: Visualizations and analysis plots

## Dataset Structure

The project expects the BreakHis dataset in this structure:
```
data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/
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

Each patient directory contains subdirectories for different magnifications (40X, 100X, 200X, 400X).

## Enhanced Features for 96% Accuracy

### Advanced Dataset Utilization (`preprocess/multimagset.py`)
- **AdvancedMultiMagDataset**: Uses all available images per patient instead of random sampling
- **Multiple sampling strategies**: `all_images`, `balanced_per_patient`, `single_per_patient`
- **Sophisticated augmentation**: Histopathology-specific transforms with 3 levels (low/medium/high)
- **Mixup augmentation**: Data mixing for better generalization
- **Smart fallback**: Handles missing magnifications gracefully

### Anti-Overfitting Training Pipeline (`training/enhanced_train_k_fold.py`)
- **Label smoothing**: Prevents overconfident predictions (smoothing=0.1)
- **Early stopping**: Patience-based with best weight restoration
- **Learning rate scheduling**: Cosine annealing with warm restarts
- **Gradient clipping**: Prevents gradient explosion (max_norm=1.0)
- **Weight decay**: L2 regularization (0.01)
- **Progressive dropout**: Dynamically adjusts dropout during training

### Test-Time Augmentation (TTA)
- **Multi-transform ensemble**: Averages predictions across 13 different augmentations
- **Flip/rotation combinations**: Geometric invariance
- **Color/brightness variations**: Robustness to staining variations
- **Multi-crop testing**: Spatial robustness

### Comprehensive Monitoring
- **Real-time progress bars**: Training and validation progress with tqdm
- **Advanced metrics**: Accuracy, balanced accuracy, precision, recall, F1, AUC
- **Learning rate tracking**: Monitor LR schedule effectiveness
- **Training history**: Complete logs saved per fold
- **Visualization ready**: All metrics structured for plotting

## Important Implementation Details

### Environment Detection
- `utils/env.py`: Handles RunPod vs local environment detection
- Automatically adjusts paths and worker counts based on environment

### Model Features
- Input format: Dictionary with keys `{'mag_40', 'mag_100', 'mag_200', 'mag_400'}`
- Supports attention map extraction for visualization
- Built-in model info reporting for parameter counting
- Optional feature extraction mode with `return_features=True`

### Training Features
- Patient-wise K-fold splitting prevents data leakage
- Weighted loss for handling class imbalance
- Best model selection based on validation balanced accuracy
- Comprehensive metrics logging and visualization

### Key Dependencies
- PyTorch/TorchVision for deep learning
- scikit-learn for metrics and splitting
- tqdm for progress bars
- numpy for numerical operations
- PIL for image processing
- matplotlib/seaborn for visualization (optional)

## Testing and Validation
- K-fold cross-validation with patient-wise splitting
- Enhanced validation with comprehensive metrics
- Test-time augmentation for maximum performance
- Model evaluation: accuracy, balanced accuracy, precision, recall, F1-score, AUC

## Performance Optimization Tips
1. **Use enhanced training**: `python main_enhanced.py` for best results
2. **Monitor validation**: Watch for overfitting with early stopping
3. **TTA for final test**: Always use test-time augmentation for best accuracy
4. **Batch size tuning**: Reduce if memory issues, increase if underutilizing GPU
5. **Learning rate**: Start with 2e-4, adjust based on convergence patterns