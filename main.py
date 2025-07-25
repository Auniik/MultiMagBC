"""
MMNet - Multi-Magnification Network for BreakHis Dataset
Main entry point for data analysis and training
"""

import os
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from backbones.our.model import MMNet
from config import (SLIDES_PATH, LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE, 
                    LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR, DROPOUT_RATE, WEIGHT_DECAY,
                    FOCAL_ALPHA, FOCAL_GAMMA, LABEL_SMOOTHING, MIXUP_ALPHA, FocalLoss, 
                    get_training_config, calculate_class_weights, mixup_data, mixup_criterion)
from preprocess.kfold_splitter import PatientWiseKFoldSplitter
import torchvision.transforms as T
from torch.utils.data import DataLoader


from preprocess.multimagset import MultiMagPatientDataset
from training.train_mm_k_fold import eval_model, eval_model_with_threshold_optimization, train_one_epoch
from sklearn.model_selection import train_test_split



def main():
    print("MMNet - Multi-Magnification Network for Breast Cancer Classification")

    from utils.helpers import seed_everything
    config = get_training_config()
    device = config['device']
    seed_everything(config['random_seed'])
    
    print(f"Using device: {device}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    print("\nDataset Analysis:")
    from preprocess.analyze import analyze_dataset
    analyze_dataset()

    # Initialize splitter
    splitter = PatientWiseKFoldSplitter(
        dataset_dir=SLIDES_PATH,
        n_splits=5,
        stratify_subtype=False
    )
    splitter.print_summary()
    patient_dict = splitter.patient_dict

    # Enhanced data augmentation for medical images
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),  # Medical images can be rotated
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
        T.RandomApply([T.ElasticTransform(alpha=50.0, sigma=5.0)], p=0.2),  # Simulate tissue deformation
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        T.RandomErasing(p=0.1, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random')  # Occlusion
    ])
    eval_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    fold_metrics = []
    for fold_idx, (train_pats, test_pats) in enumerate(splitter.folds):
        print(f"\n===== Fold {fold_idx} =====")
        print(f"Train patients: {len(train_pats)}, Test patients: {len(test_pats)}")
        # Datasets with refined multi-image sampling (less aggressive)
        train_ds = MultiMagPatientDataset(
            patient_dict, train_pats, transform=train_transform,
            samples_per_patient=3,  # Reduced from 5 to prevent overfitting
            adaptive_sampling=True  # Use adaptive strategy based on available images
        )
        test_ds = MultiMagPatientDataset(
            patient_dict, test_pats, transform=eval_transform,
            samples_per_patient=1,  # Keep single sample for consistent evaluation
            adaptive_sampling=False
        )
        
        # Print sampling statistics
        train_stats = train_ds.get_sampling_stats()
        print(f"Training samples per epoch: {train_stats['total_samples_per_epoch']} "
              f"(avg utilization: {train_stats['avg_utilization']:.1%})")
        
        test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, 
                               num_workers=config['num_workers'], pin_memory=config['pin_memory'])

        # Calculate class weights for imbalanced dataset
        train_labels = [train_ds.patient_dict[pid]['label'] for pid in train_pats]
        class_weights = calculate_class_weights(train_labels).to(device)
        print(f"Class weights: Benign={class_weights[0]:.2f}, Malignant={class_weights[1]:.2f}")
        
        # Model, criterion, optimizer, scheduler
        epochs = NUM_EPOCHS

        model = MMNet(dropout=DROPOUT_RATE).to(device)
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, weight=class_weights, label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=LR_SCHEDULER_FACTOR, 
            patience=LR_SCHEDULER_PATIENCE
        )

        # Split training data into train/validation for nested CV (increased validation size)
        train_pats_inner, val_pats = train_test_split(train_pats, test_size=0.3, random_state=42, stratify=[train_ds.patient_dict[pid]['label'] for pid in train_pats])
        
        # Create inner validation dataset (single sample for consistent validation)
        val_ds = MultiMagPatientDataset(
            patient_dict, val_pats, transform=eval_transform,
            samples_per_patient=1, adaptive_sampling=False
        )
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, 
                              num_workers=config['num_workers'], pin_memory=config['pin_memory'])
        
        # Update training loader with reduced training set and refined sampling
        train_ds_inner = MultiMagPatientDataset(
            patient_dict, train_pats_inner, transform=train_transform,
            samples_per_patient=3,  # Reduced base samples per patient
            adaptive_sampling=True
        )
        
        # Adjust batch size for increased data volume
        inner_train_stats = train_ds_inner.get_sampling_stats()
        samples_per_epoch = inner_train_stats['total_samples_per_epoch']
        
        # Dynamic batch size adjustment based on data volume
        if samples_per_epoch > 2000:
            # Large dataset: keep original batch size
            effective_batch_size = config['batch_size']
        elif samples_per_epoch > 1000:
            # Medium dataset: slight increase
            effective_batch_size = min(config['batch_size'] + 4, 32)
        else:
            # Small dataset: increase batch size more
            effective_batch_size = min(config['batch_size'] + 8, 32)
            
        print(f"Inner training samples: {samples_per_epoch}, batch size: {effective_batch_size}")
        
        train_loader_inner = DataLoader(
            train_ds_inner, batch_size=effective_batch_size, shuffle=True, 
            num_workers=config['num_workers'], pin_memory=config['pin_memory'], 
            drop_last=True
        )
        
        print(f"Inner split: Train {len(train_pats_inner)}, Val {len(val_pats)} patients")
        
        best_val_bal_acc = 0
        epochs_no_improve = 0
        best_model_state = None
        optimal_threshold = 0.5
        
        # For overfitting detection
        train_losses, val_losses = [], []
        overfitting_patience = 5
        overfitting_threshold = 0.1  # If val_loss > train_loss + threshold for multiple epochs
        
        for epoch in range(1, epochs+1):
            # Set epoch for deterministic sampling diversity
            train_ds_inner.set_epoch(epoch)
            train_loss, train_acc = train_one_epoch(
                model, train_loader_inner, criterion, optimizer, device, 
                use_mixup=True, mixup_alpha=MIXUP_ALPHA
            )
            val_loss, val_acc, val_bal, val_f1, val_auc, threshold = eval_model_with_threshold_optimization(
                model, val_loader, criterion, device, use_dropout=True
            )
            
            # Track losses for overfitting detection
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Step scheduler with validation balanced accuracy
            scheduler.step(val_bal)
            
            # Overfitting detection
            overfitting_warning = ""
            perfect_validation_warning = ""
            
            # Check for perfect validation performance (sign of overfitting)
            if val_bal >= 0.995 or val_auc >= 0.995:
                perfect_validation_warning = " [PERFECT VAL - POSSIBLE OVERFITTING]"
            
            # Check for train-validation loss divergence
            if len(train_losses) >= overfitting_patience:
                recent_train_loss = np.mean(train_losses[-overfitting_patience:])
                recent_val_loss = np.mean(val_losses[-overfitting_patience:])
                if recent_val_loss > recent_train_loss + overfitting_threshold:
                    overfitting_warning = " [TRAIN-VAL DIVERGENCE]"
            
            print(f"Epoch {epoch:02d}: "
                  f"Train: Loss {train_loss:.4f}, Acc {train_acc:.3f} |"
                  f"Val: Loss {val_loss:.4f}, Acc {val_acc:.3f}, "
                  f"BalAcc {val_bal:.3f}, F1 {val_f1:.3f}, AUC {val_auc:.3f}, Thresh {threshold:.3f}"
                  f"{overfitting_warning}{perfect_validation_warning}")
            
            # Save best model and implement early stopping
            if val_bal > best_val_bal_acc:
                best_val_bal_acc = val_bal
                best_model_state = model.state_dict().copy()
                optimal_threshold = threshold
                epochs_no_improve = 0
                print(f"New best validation balanced accuracy: {best_val_bal_acc:.3f}, threshold: {optimal_threshold:.3f}")
            else:
                epochs_no_improve += 1
            
            # Early stopping
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {epoch} epochs (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break
        
        # Load best model and evaluate on test set with optimized threshold
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            ckpt_path = os.path.join(config['output_dir'], 'models', f"best_model_fold_{fold_idx}.pth")
            torch.save(best_model_state, ckpt_path)
            print(f"Best model saved: {ckpt_path} (Val BalAcc: {best_val_bal_acc:.3f})")
        
        # Final test evaluation with optimized threshold (NO threshold optimization on test set)
        _, test_acc, test_bal, test_f1, test_auc, _ = eval_model(model, test_loader, criterion, device, optimal_threshold)
        print(f"Test Results: Acc {test_acc:.3f}, BalAcc {test_bal:.3f}, F1 {test_f1:.3f}, AUC {test_auc:.3f} (threshold: {optimal_threshold:.3f})")
        
        fold_metrics.append((test_acc, test_bal, test_f1, test_auc))
    # Summary
    accs, bals, f1s, aucs = zip(*fold_metrics)
    print("\n=== Cross-Validation Results ===")
    print(f"Acc: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"BalAcc: {np.mean(bals):.3f} ± {np.std(bals):.3f}")
    print(f"F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")



if __name__ == "__main__":
    main()