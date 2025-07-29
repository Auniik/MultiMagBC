#!/usr/bin/env python3
"""
Training script for MultiMagLightweightCNN
Integrates with existing BreakHis training pipeline
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

from backbones.our.model import MultiMagLightweightCNN

# Import existing components from your codebase
from config import (
    SLIDES_PATH, LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
    LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR, WEIGHT_DECAY,
    FocalLoss, get_training_config, calculate_class_weights
)
from preprocess.kfold_splitter import PatientWiseKFoldSplitter
from preprocess.multimagset import MultiMagPatientDataset
from training.train_mm_k_fold import train_one_epoch, eval_model_with_threshold_optimization, eval_model
import torchvision.transforms as T


def create_transforms():
    """Create data augmentation transforms optimized for lightweight model"""
    
    # Lighter augmentation for small model
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=10),  # Reduced from 15
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),  # Reduced
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),  # Reduced
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    eval_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, eval_transform


def train_lightweight_model():
    """Main training function for lightweight model"""
    
    print("=== Training MultiMagLightweightCNN ===")
    print("Lightweight architecture with ~300K parameters\n")
    
    # Setup
    from utils.helpers import seed_everything
    config = get_training_config()
    device = config['device']
    seed_everything(config['random_seed'])
    
    print(f"Using device: {device}")
    
    # Initialize data splitter
    splitter = PatientWiseKFoldSplitter(
        dataset_dir=SLIDES_PATH,
        n_splits=5,
        stratify_subtype=False
    )
    patient_dict = splitter.patient_dict
    
    # Create transforms
    train_transform, eval_transform = create_transforms()
    
    # Hyperparameters optimized for lightweight model
    LIGHTWEIGHT_CONFIG = {
        'base_channels': 24,  # Can increase to 32 for more capacity
        'dropout': 0.3,       # Less dropout needed for smaller model
        'learning_rate': 1e-4,  # Higher LR for faster convergence
        'weight_decay': 1e-4,   # Less weight decay
        'label_smoothing': 0.1,  # Less smoothing
        'mixup_alpha': 0.1,      # Less mixup
        'focal_gamma': 2.0,
        'focal_alpha': 0.7
    }
    
    fold_metrics = []
    
    for fold_idx, (train_pats, test_pats) in enumerate(splitter.folds):
        print(f"\n===== Fold {fold_idx} =====")
        print(f"Train patients: {len(train_pats)}, Test patients: {len(test_pats)}")
        


        train_ds = MultiMagPatientDataset(patient_dict, train_pats, transform=train_transform, mode='train')
        # val_ds = MultiMagPatientDataset(patient_dict, val_pats, transform=eval_transform, mode='val', full_utilization_mode='all')
        test_ds = MultiMagPatientDataset(patient_dict, test_pats, transform=eval_transform, mode='test', full_utilization_mode='all')

    
        
        # Print dataset info
        train_stats = train_ds.get_sampling_stats()
        print(f"Training stats: {train_stats}")
        
        # Create data loaders
        test_loader = DataLoader(
            test_ds, batch_size=config['batch_size'], 
            shuffle=False, num_workers=config['num_workers']
        )
        
        # Calculate class weights
        train_labels = [train_ds.patient_dict[pid]['label'] for pid in train_pats]
        class_weights = calculate_class_weights(train_labels).to(device)
        print(f"Class weights: Benign={class_weights[0]:.2f}, Malignant={class_weights[1]:.2f}")
        
        # Initialize model
        model = MultiMagLightweightCNN(
            num_classes=2,
            base_channels=LIGHTWEIGHT_CONFIG['base_channels'],
            dropout=LIGHTWEIGHT_CONFIG['dropout']
        ).to(device)
        
        # Print model info for first fold
        if fold_idx == 0:
            model_info = model.get_model_info()
            print(f"\nModel parameters: {model_info['total_parameters']:,}")
            print(f"Breakdown:")
            for key, value in model_info.items():
                if key != 'total_parameters':
                    print(f"  {key}: {value:,}")
        
        # Loss function
        criterion = FocalLoss(
            alpha=LIGHTWEIGHT_CONFIG['focal_alpha'],
            gamma=LIGHTWEIGHT_CONFIG['focal_gamma'],
            weight=class_weights,
            label_smoothing=LIGHTWEIGHT_CONFIG['label_smoothing']
        )
        
        # Optimizer - AdamW with decoupled weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=LIGHTWEIGHT_CONFIG['learning_rate'],
            weight_decay=LIGHTWEIGHT_CONFIG['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
        )
        
        # Split for validation
        train_pats_inner, val_pats = train_test_split(
            train_pats, test_size=0.2,  # 20% validation instead of 30%
            random_state=42,
            stratify=[train_ds.patient_dict[pid]['label'] for pid in train_pats]
        )
        
        # Create validation dataset
        val_ds = MultiMagPatientDataset(
            patient_dict, val_pats, transform=eval_transform,
            #samples_per_patient=1, #adaptive_sampling=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=config['batch_size'],
            shuffle=False, num_workers=config['num_workers']
        )
        
        # Update training dataset
        train_ds_inner = MultiMagPatientDataset(
            patient_dict, train_pats_inner, transform=train_transform,
            #samples_per_patient=2, #adaptive_sampling=True
        )
        train_loader_inner = DataLoader(
            train_ds_inner, batch_size=config['batch_size'],
            shuffle=True, num_workers=config['num_workers'],
            drop_last=True
        )
        
        print(f"Inner split: Train {len(train_pats_inner)}, Val {len(val_pats)} patients")
        
        # Training loop
        best_val_bal_acc = 0
        epochs_no_improve = 0
        best_model_state = None
        optimal_threshold = 0.5
        
        for epoch in range(1, NUM_EPOCHS + 1):
            # Set epoch for sampling diversity
            train_ds_inner.set_epoch(epoch)
            
            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader_inner, criterion, optimizer, device,
                use_mixup=True, mixup_alpha=LIGHTWEIGHT_CONFIG['mixup_alpha']
            )
            
            # Validate with threshold optimization
            val_loss, val_acc, val_bal, val_f1, val_auc, prec, rec, threshold = eval_model_with_threshold_optimization(
                model, val_loader, criterion, device, use_dropout=False  # No dropout for lightweight
            )
            
            # Update scheduler
            scheduler.step(val_bal)
            
            print(f"Epoch {epoch:02d}: "
                  f"Train: Loss {train_loss:.4f}, Acc {train_acc:.3f} | "
                  f"Val: Loss {val_loss:.4f}, Acc {val_acc:.3f}, "
                  f"BalAcc {val_bal:.3f}, P {prec:.3f}, R {rec:.3f}, F1 {val_f1:.3f}, AUC {val_auc:.3f}")
            
            # Save best model
            if val_bal > best_val_bal_acc:
                best_val_bal_acc = val_bal
                best_model_state = model.state_dict().copy()
                optimal_threshold = threshold
                epochs_no_improve = 0
                print(f"  → New best validation balanced accuracy: {best_val_bal_acc:.3f}")
            else:
                epochs_no_improve += 1
            
            # Early stopping
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {epoch} epochs")
                break
        
        # Load best model and save
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            save_path = os.path.join(
                config['output_dir'], 'models', 
                f'lightweight_model_fold_{fold_idx}.pth'
            )
            torch.save({
                'model_state_dict': best_model_state,
                'config': LIGHTWEIGHT_CONFIG,
                'fold': fold_idx,
                'optimal_threshold': optimal_threshold,
                'val_metrics': {
                    'balanced_accuracy': best_val_bal_acc,
                }
            }, save_path)
            print(f"Best model saved: {save_path}")
        
        # Test evaluation
        eval_history = eval_model(
            model, test_loader, criterion, device, optimal_threshold
        )
        print(f"Test Results: Acc {eval_history['accuracy']:.3f}, BalAcc {eval_history['balanced_accuracy']:.3f}, "
              f"F1 {eval_history['f1_score']:.3f}, AUC {eval_history['auc']:.3f}")

        fold_metrics.append((eval_history['accuracy'], eval_history['balanced_accuracy'],
                             eval_history['f1_score'], eval_history['auc']))

        # Get attention maps for visualization (optional)
        if fold_idx == 0:  # Only for first fold
            print("\nExtracting attention maps for visualization...")
            model.eval()
            with torch.no_grad():
                # Get one batch for visualization
                images_dict, labels = next(iter(test_loader))
                images_dict = {k: v.to(device) for k, v in images_dict.items()}
                
                # Get attention maps
                attention_data = model.get_attention_maps(images_dict)
                fusion_weights = attention_data['fusion_weights'][0].cpu().numpy()
                
                print(f"Magnification importance weights:")
                for i, mag in enumerate(['40x', '100x', '200x', '400x']):
                    print(f"  {mag}: {fusion_weights[i]:.3f}")
    
    # Summary
    accs, bals, f1s, aucs = zip(*fold_metrics)
    print("\n=== Cross-Validation Results (Lightweight Model) ===")
    print(f"Accuracy:  {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"Balanced:  {np.mean(bals):.3f} ± {np.std(bals):.3f}")
    print(f"F1 Score:  {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"AUC:       {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"Total folds: {len(fold_metrics)}")

    
    return fold_metrics


if __name__ == "__main__":
    train_lightweight_model()