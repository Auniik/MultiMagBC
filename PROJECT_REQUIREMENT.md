Deep Learning CNN Model Implementation PRD - MMNet for BreakHis Dataset

Executive Summary
This PRD outlines the implementation of MMNet (Multi-Magnification Network), a novel CNN architecture designed to reduce pathologist workflow by leveraging multi-scale histopathology images from the BreakHis dataset. The model incorporates spatial attention, channel attention, hierarchical magnification fusion, and cross-magnification learning to achieve state-of-the-art performance in breast cancer histopathology classification.

1. Project Overview
1.1 Objective
Develop a lightweight, interpretable CNN model that:

Achieves 96-98% accuracy on binary classification (benign/malignant)
Effectively handles multi-magnification inputs (40x, 100x, 200x, 400x)
Provides interpretable attention mechanisms
Reduces computational requirements compared to existing approaches

1.2 Key Innovations
Hierarchical Magnification Attention (HMA) mechanism
Cross-Magnification Fusion (CMF) with learned importance weights
Multi-Scale Attention Pooling (MSAP)
Dual classification heads for binary and subtype classification


2. Technical Requirements
2.1 Model Architecture Components
2.1.1 Spatial Attention Module

Apply within each magnification branch
Focus on diagnostically relevant regions
Implementation: Try depthwise conv layers for lightweightness

2.1.2 Channel Attention Module
Learn feature importance across channels


2.1.3 Hierarchical Magnification Attention (HMA)

Progressive fusion: 40x → 100x → 200x → 400x
Each level learns to incorporate lower magnification context
Attention weights for each magnification level


2.1.4 Cross-Magnification Fusion (CMF)
Learnable importance weights for each magnification
Dynamic weighting based on input quality
Fusion strategy: weighted concatenation with attention

2.1.5 Multi-Scale Attention Pooling (MSAP)
Multiple pooling scales (1x1, 2x2, 4x4)
Attention-weighted aggregation
Preserves both global and local features


2.3 Data Handling Strategy
2.3.1 Imbalance Handling

Binary Classification: Class-weighted focal loss
Subtype Classification:
Group rare subtypes (< 500 images) into "Other" category
Use class-balanced cross-entropy
Implement SMOTE for severe imbalance


2.3.2 Cross-Validation Strategy
Primary: Patient-wise 5-fold CV (ensure no patient leak)
Secondary: Fixed 70/15/15 train/val/test split for deployment simulation
Stratification: Maintain class distribution across folds


4. Evaluation Metrics & Deliverables
4.1 Performance Metrics

Binary Classification: Accuracy, Precision, Recall, F1-Score, AUC-ROC
Subtype Classification: Per-class metrics, Macro/Micro averages
Confidence Analysis: Calibration plots, confidence distributions
Magnification Importance: Learned attention weights visualization

4.2 Visualizations
Confusion Matrices: Binary and multi-class
ROC Curves: With AUC values
Attention Heatmaps: Spatial attention visualization
Grad-CAM: Class activation maps for interpretability
Magnification Importance: Bar plots of learned weights
Training Curves: Loss, accuracy, learning rate

4.3 Tables
Cross-Validation Results (Table 1)
Fixed Split Results (Table 2)
Ablation Study Results (Table 3)
Baseline Comparisons (Table 4)
Per-Magnification Performance (Table 5)

5. Ablation Studies
5.1 Components to Ablate
Without Spatial Attention
Without Channel Attention
Without Hierarchical Attention
Without Cross-Magnification Fusion
Single Magnification Only (40x, 100x, 200x, 400x separately)
Simple Concatenation Baseline

5.2 Ablation Metrics

Performance drop analysis
Computational efficiency (FLOPs, parameters)
Inference time comparison

6. Baseline Comparisons
6.1 Baseline Models
Simple Concatenation: Direct feature concatenation
Single Magnification: Best performing single magnification

6.2 Comparison Metrics
Accuracy improvement
Parameter efficiency
Inference speed

7. Deployment Configuration
7.1 RunPod Setup (setup.sh)

8. Success Criteria
8.1 Primary Metrics

Binary classification accuracy: 96-98%
F1-Score: > 0.95
AUC-ROC: > 0.98

8.2 Secondary Metrics

Training time: < 30mins on single GPU
Inference time: < 50ms per patient
Interpretability: Clear attention visualizations

9. Risk Mitigation
9.1 Technical Risks

Overfitting: Addressed via augmentation, dropout, early stopping
Class Imbalance: Focal loss, class weighting, SMOTE, or any other
Memory Constraints: Gradient accumulation, mixed precision training

9.2 Performance Risks
Below Target Accuracy: Ensemble methods, hyperparameter tuning
Computational Overhead: Model pruning, knowledge distillation

10. Timeline & Milestones
Phase 1: Foundation (Days 1-3)
Environment setup
Data analysis and preprocessing
Basic model implementation

Phase 2: Core Development (Days 4-7)
Attention mechanisms implementation
Cross-magnification fusion
Training pipeline

Phase 3: Optimization (Days 8-10)
Hyperparameter tuning
Ablation studies
Performance optimization

Phase 4: Evaluation (Days 11-12)
Comprehensive evaluation
Visualization generation
Documentation

11. Output Structure
All results will be exported to the output/ directory:
output/
├── tables/
│   ├── table_1_cross_validation_k_fold_results.csv
│   ├── table_2_patient_wise_results.csv
│   ├── table_3_ablation_results.csv
│   └── table_4_baseline_comparison.csv
├── plots/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   ├── attention_maps/
│   ├── gradcam/
│   └── training_curves/
├── models/
│   ├── best_model.pth
│   └── checkpoints/

12.2 Results Documentation
RUNPOD_OUTPUT.md: User will update this file with the output from Runpod pipeline run
------
=== Dataset Summary ===
Total Patients: 82
Benign Patients: 24
Malignant Patients: 58
Total Images: 7909
Images per Class: {'malignant': 5429, 'benign': 2480}
Images per Magnification: {'100X': 2081, '400X': 1820, '40X': 1995, '200X': 2013}
Images per Magnification per Class:
       malignant  benign
100X       1437     644
400X       1232     588
40X        1370     625
200X       1390     623
Top Tumor Subtypes:
          Tumor Subtype  Patients  Total Images
2     ductal_carcinoma        38          3451
6         fibroadenoma        10          1014
0   mucinous_carcinoma         9           792
3    lobular_carcinoma         5           626
5      tubular_adenoma         7           569
1  papillary_carcinoma         6           560
7      phyllodes_tumor         3           453
4             adenosis         4           444

Project Architechture:
├── backbones
│   ├── __init__.py
│   ├── baseline_simple_concat.py
│   ├── baseline_single.py
│   └── our
│       ├── gradcam.py # Grad-Cam related all codes
│       └── model.py # Our model MMNet with all the supportive classes
├── config.py # Training Configuration, Base Paths, 
├── main.py # main file to run the complete pipeline.
├── output # all the plottings and tables will be exported here
├── preprocess
│   ├── analyze.py # data understanding functions
│   └── preprocess.py # preprocessing, augmentations etc
├── PROJECT_REQUIREMENT.md # PRD document
├── requirements.runpod # it will be run after starting on demand runpod instance
├── requirements.txt # necessary files
├── RUNPOD_OUTPUT.md # After running pipeline in the runpod instance the output will be printed here to analyze
├── setup.sh # this will be run in Runpod after on demand instance deploy
├── tests
│   └── analyze_dataset.py # all the test/debug files will be here
└── utils
    ├── env.py # environment related functions
    ├── plottings.py # All the plotting related codes will be there
    ├── tables.py # All the table related code will be there
    └── helpers.py # helper functions

CRITICAL NOTE: In code implementation writing less comment will be good. Make cleaner, minimal indentation, functional and quality approach not jargons.