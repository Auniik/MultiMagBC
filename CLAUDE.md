# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
MMNet (Multi-Magnification Network) for BreakHis breast cancer histopathology classification. A CNN architecture with spatial/channel attention, hierarchical magnification fusion, and cross-magnification learning targeting 96-98% accuracy on binary classification (benign/malignant) across 4 magnification levels (40x, 100x, 200x, 400x).

## Development Commands

### Environment Setup
```bash
# RunPod environment (primary deployment target)
bash setup.sh

# Install dependencies
pip install -r requirements.runpod  # For RunPod (PyTorch 2.1.0 + CUDA 11.8)
pip install -r requirements.txt     # For local development
```

### Running the Pipeline
```bash
# Main training pipeline
python main.py

# Dataset analysis only
python preprocess/analyze.py

# Test/debug utilities
python tests/analyze_dataset.py
```

### Monitoring
```bash
# GPU monitoring during training
watch -n 1 nvidia-smi
```

## Architecture Overview

### Project Structure
Arch example:
├── backbones
├── __init__.py
├── baseline_simple_concat.py
├── baseline_single.py
└── our
│     ├── gradcam.py # Grad-Cam related all codes
│     └── model.py # Our model MMNet with all the supportive classes
├── config.py # Training Configuration, Base Paths, 
├── main.py # main file to run the complete pipeline.
├── output # all the plottings and tables will be exported here
├── preprocess
│ ├── analyze.py # data understanding functions
│ └── preprocess.py # preprocessing, augmentations etc
├── PROJECT_REQUIREMENT.md # PRD document
├── requirements.runpod # it will be run after starting on demand runpod instance
├── requirements.txt # necessary files
├── RUNPOD_OUTPUT.md # After running pipeline in the runpod instance the output will be printed here to analyze
├── setup.sh # this will be run in Runpod after on demand instance deploy
├── tests
│ └── analyze_dataset.py # all the test/debug files will be here
└── utils
    ├── env.py # environment related functions
    ├── plottings.py # All the plotting related codes will be there
    ├── tables.py # All the table related code will be there
    └── helpers.py # helper functions
```

### Key Components

#### MMNet Architecture (backbones/our/model.py)
- **Spatial Attention Module**: Depthwise convolutions for lightweight region focus
- **Channel Attention Module**: Feature importance learning across channels  
- **Hierarchical Magnification Attention (HMA)**: Progressive fusion 40x→100x→200x→400x
- **Cross-Magnification Fusion (CMF)**: Learnable importance weights with dynamic weighting
- **Multi-Scale Attention Pooling (MSAP)**: Multiple pooling scales (1x1, 2x2, 4x4)
- **Dual Classification Heads**: Binary (benign/malignant) + subtype classification

#### Configuration System (config.py)
- Auto-detects environment: CUDA/MPS/CPU
- Device-specific batch sizes and optimization settings
- Multi-GPU support with DataParallel
- Tensor core optimization for A100/V100/RTX GPUs

#### Data Handling Strategy
- **Patient-wise 5-fold CV**: Prevents patient data leakage
- **Class imbalance handling**: Focal loss for binary, class-weighted CE for subtypes
- **Multi-magnification input**: Handles 4 magnification levels per patient
- **BreakHis dataset**: 82 patients, 7909 images across benign/malignant classes

## Development Guidelines

### Model Implementation
- MMNet class in `backbones/our/model.py` contains the complete architecture
- Follow the attention mechanism patterns established in the baseline
- Use `timm` library for backbone CNN architectures
- Implement grad-CAM visualization in `backbones/our/gradcam.py`

### Training Configuration
- Environment auto-detection via `get_training_config()` in config.py
- RunPod optimized: PyTorch 2.1.0 + CUDA 11.8 + NumPy 1.24.3 compatibility
- Multi-GPU training automatically enabled when available
- Use `seed_everything(42)` for reproducibility

### Data Paths and Environment
- Base paths configured via `utils/env.py` and `get_base_path()`
- BreakHis data expected in `data/breakhis/` directory
- Fold information in `Folds.csv` for cross-validation splits
- Output results go to `output/` with subdirectories for tables, plots, models

### Results and Evaluation
All outputs structured in `output/` directory and stdout for tables and train/val/test and other logs:
- `tables/`: Metrics, CSV files for cross-validation, ablation studies, baselines
- `plots/`: Confusion matrices, ROC curves, attention maps, Grad-CAM
- `models/`: Best model checkpoints and training snapshots

### Performance Targets
- **Binary Classification**: 96-98% accuracy (Balanced and Standard), F1 > 0.95, AUC-ROC > 0.98
- **Training Time**: < 30 minutes on single GPU
- **Inference**: < 50ms per patient
- **Interpretability**: Clear attention visualizations and Grad-CAM outputs

### Ablation Studies Required
Test performance without: spatial attention, channel attention, hierarchical attention, cross-magnification fusion, and single magnification baselines.

## Environment Compatibility
- **Primary**: RunPod with PyTorch 2.14.0 + CUDA
- **Secondary**: Local development with MPS (Apple Silicon) or CPU fallback
- **Dependencies**: Optimized requirements files for each environment
- **GPU Memory**: Configured for efficient multi-GPU training with gradient accumulation

Note: Don't do too much expensive garbage code. try to make the implementation cleaner.