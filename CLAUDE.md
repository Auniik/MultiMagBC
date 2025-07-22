# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MMNet is a Multi-Magnification Network for breast cancer histopathology classification using the BreakHis dataset. The system processes microscopy images at four magnification levels (40x, 100x, 200x, 400x) simultaneously using advanced attention mechanisms for improved classification accuracy.

## Key Commands

### Setup and Environment
- **Initial setup**: `bash setup.sh` - Sets up RunPod environment with proper CUDA/PyTorch configuration
- **Install dependencies**: `pip install -r requirements.txt` (local) or `pip install -r requirements.runpod` (RunPod environment)

### Training and Evaluation
- **Main training**: `python main.py` - Runs full 5-fold cross-validation training pipeline
- **Single magnification training**: `python training/train_single_mag.py`
- **Simple concatenation baseline**: `python training/train_simple_concat.py`

### Testing
- **Run tests**: `python -m pytest tests/` or `pytest tests/`
- **Specific test**: `python -m pytest tests/test_analyze_dataset.py`

### Data Analysis
- **Dataset analysis**: The analysis is automatically run when executing `main.py`, or can be run separately by importing `preprocess.analyze.analyze_dataset()`

### Result Analysis
- **Training results**: Check RUNPOD_OUTPUT.md for detailed training logs and cross-validation results

## Architecture Overview

### Core Components

1. **MMNet Model** (`backbones/our/model.py`):
   - Multi-magnification network with hierarchical attention
   - Processes 4 magnifications simultaneously: 40x, 100x, 200x, 400x
   - Uses EfficientNet-B1 as backbone by default
   - Features advanced attention mechanisms for spatial, channel, and cross-magnification fusion

2. **Attention Mechanisms** (`backbones/our/attention.py`):
   - `MultiScaleAttentionPool`: Multi-scale spatial attention
   - `HierarchicalMagnificationAttention`: Learns relationships between magnification levels
   - `ClinicalChannelAttention`: Enhanced channel attention with dual pooling
   - `ClinicalCrossMagFusion`: Cross-magnification fusion with attention

3. **Data Pipeline**:
   - `preprocess/multimagset.py`: Enhanced dataset class with adaptive multi-image sampling per patient
     - **Adaptive Sampling Strategy**: Low-volume patients (≤15 imgs/mag) use ~60% of images, medium-volume (16-30) ~40%, high-volume (>30) ~25%
     - **Multi-Image Support**: Samples 3 base images per patient with conservative utilization to prevent overfitting (~3-4x increase in samples per epoch)
     - **Deterministic Sampling**: Uses epoch-based seeding for reproducible yet diverse sampling across epochs
     - **Dynamic Batch Sizing**: Automatically adjusts batch sizes based on effective dataset size
   - `preprocess/kfold_splitter.py`: Patient-wise K-fold splitting to prevent data leakage
   - `preprocess/analyze.py`: Comprehensive dataset analysis and statistics

4. **Training Framework**:
   - `training/train_mm_k_fold.py`: Training and evaluation functions for multi-magnification model
   - 5-fold cross-validation with patient-wise splitting
   - Uses AdamW optimizer with cosine annealing learning rate schedule

### Data Structure

The dataset expects the BreakHis v1 structure:
```
data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/
   benign/SOB/[tumor_type]/[patient_id]/[magnification]/
   malignant/SOB/[tumor_type]/[patient_id]/[magnification]/
```

Magnifications: 40X, 100X, 200X, 400X
Classes: Benign (0), Malignant (1)
Tumor subtypes: 8 categories (adenosis, fibroadenoma, phyllodes_tumor, tubular_adenoma, ductal_carcinoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma)
using preprocess/kfold_splitter.py file,
got the === Fold-wise Dataset Summary ===

### Configuration

- **Device detection**: Automatically detects CUDA, MPS, or CPU (`config.py:get_device()`)
- **Training config**: Adaptive batch sizes and worker counts based on device (`config.py:get_training_config()`)
- **Hyperparameters**: Configurable in `config.py` (learning rate: 5e-5, epochs: 25, image size: 224x224)
- **Enhanced regularization**: Dropout 0.7, weight decay 5e-3, label smoothing 0.2, mixup augmentation α=0.2
- **Focal loss**: Alpha=0.7, gamma=2.0 for better class imbalance handling
- **Nested cross-validation**: 30% validation split with validation dropout for proper regularization
- **Early stopping**: Patience of 7 epochs based on validation balanced accuracy
- **Learning rate scheduling**: ReduceLROnPlateau with patience of 3 epochs
- **Overfitting detection**: Monitors train/validation loss gap and perfect validation performance with warnings

## Key Features

### Multi-Magnification Processing
The model processes all 4 magnifications simultaneously, learning hierarchical relationships where higher magnifications can attend to lower magnifications for context.

### Advanced Attention Mechanisms
- **Spatial Attention**: Focuses on important regions within each magnification
- **Channel Attention**: Learns importance of different feature channels
- **Hierarchical Attention**: Models magnification hierarchy (40x�100x�200x�400x)
- **Cross-Magnification Fusion**: Combines information across all magnifications

### Patient-Wise Cross-Validation
Ensures no data leakage by splitting patients (not images) across folds, maintaining realistic evaluation scenarios.

### Comprehensive Evaluation
- **Nested cross-validation**: Inner validation loop prevents threshold overfitting on test set
- **Robust threshold optimization**: Optimized on validation set, applied to test set
- **Enhanced data augmentation**: Rotation, elastic transforms, blur, color jitter, random erasing
- **Focal loss with label smoothing**: Better class balance and reduced overconfidence  
- **Overfitting monitoring**: Real-time detection of train/validation loss divergence
- **Gradient clipping**: Prevents exploding gradients with max norm of 1.0
- **Stronger regularization**: Higher dropout (0.5) and weight decay (1e-3) for better generalization

## File Organization

- `main.py`: Main entry point for training pipeline
- `config.py`: Configuration and device setup
- `backbones/our/`: Custom MMNet architecture and attention modules  
- `preprocess/`: Data preprocessing, analysis, and K-fold splitting
- `training/`: Training loops and evaluation functions
- `evaluate/`: Evaluation utilities (GradCAM, plotting, tables)
- `utils/`: Helper functions and environment utilities
- `data/`: Dataset storage (BreakHis structure expected)
- `output/`: Training outputs (models, logs, plots)

## Development Notes

### Device-Specific Settings
The system automatically adjusts batch size and num_workers based on detected device:
- CUDA: batch_size=16, num_workers=8
- MPS (Apple Silicon): batch_size=8, num_workers=0  
- CPU: batch_size=4, num_workers=2

### Model Checkpoints
Best models are saved per fold as `output/fold_{i}_best.pth` based on balanced accuracy metric.

### Memory Considerations
The model processes 4 images simultaneously (one per magnification), which requires sufficient GPU memory. Batch sizes are automatically adjusted based on available hardware.

## Major Improvements
- **Enhanced Multi-Image Sampling**: Implemented conservative adaptive sampling strategy that increases data utilization from ~4% to ~15-20% per epoch, with strong regularization to prevent overfitting
- **Advanced Multi-Magnification Attention**: Implemented hierarchical attention mechanisms to learn relationships between magnification levels
- **Patient-Wise Cross-Validation**: Developed robust splitting strategy to prevent data leakage and ensure realistic model evaluation
- **Overfitting Prevention**: Comprehensive detection and prevention techniques, including early stopping, gradient clipping, and advanced regularization
- **Enhanced Data Augmentation**: Medical-specific transforms including elastic deformation, rotation, color jittering, and mixup augmentation
- **Focal Loss Implementation**: Better handling of class imbalance with enhanced label smoothing and alpha weighting
- **Strong Regularization Suite**: Increased dropout (0.7), weight decay (5e-3), validation dropout, and overfitting detection
- **Dynamic Training Pipeline**: Conservative batch size adjustment and epoch-based sampling diversity with overfitting prevention