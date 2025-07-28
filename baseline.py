import os
import json
import csv
import time
from typing import Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from backbones.baseline import SimpleConcatBaseline
from backbones.our.model1 import MMNet
from config import (SLIDES_PATH, LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE, 
                    LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR, DROPOUT_RATE, WEIGHT_DECAY,
                    SAMPLES_PER_PATIENT_BALANCED, EPOCH_MULTIPLIER_BALANCED, VAL_SAMPLES_PER_PATIENT_BALANCED,
                    FOCAL_ALPHA, FOCAL_GAMMA, LABEL_SMOOTHING, MIXUP_ALPHA, FocalLoss, 
                    get_training_config, calculate_class_weights, mixup_data, mixup_criterion)
from evaluate.gradcam import GradCAM, visualize_gradcam
from preprocess.kfold_splitter import PatientWiseKFoldSplitter

from torch.utils.data import DataLoader


from preprocess.multimagset import MultiMagPatientDataset
from preprocess.preprocess import get_transforms
from training.train_mm_k_fold import eval_model, eval_model_with_threshold_optimization, train_one_epoch

from utils.stats import save_as_json


def main():
    print("Baseline for Single mag and simple concat - Network for Breast Cancer Classification")

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

    splitter = PatientWiseKFoldSplitter(
        dataset_dir=SLIDES_PATH,
        n_splits=5,
        stratify_subtype=False
    )
    splitter.print_summary()
    patient_dict = splitter.patient_dict

    train_transform, eval_transform = get_transforms()
    for fold_idx in range(len(splitter.folds)):
        print(f"\n===== Fold {fold_idx} =====")

        train_pats, val_pats, test_pats = splitter.get_fold(fold_idx)
        print(f"Train patients: {len(train_pats)}, Val Patients: {len(val_pats)}, Test patients: {len(test_pats)}")

        train_ds = MultiMagPatientDataset(patient_dict, train_pats, transform=train_transform, mode='train')
        val_ds = MultiMagPatientDataset(patient_dict, val_pats, transform=eval_transform, mode='val', full_utilization_mode='all')
        test_ds = MultiMagPatientDataset(patient_dict, test_pats, transform=eval_transform, mode='test', full_utilization_mode='all')

        train_stats = train_ds.get_sampling_stats()
        print(f"Training samples per epoch: {train_stats}")
        print(f"Validation samples: {len(val_ds)}, Test samples: {len(test_ds)}")
        print(f"Patients with full 4 mags: {sum(1 for p in train_pats if sum(len(train_ds.patient_dict[p]['images'][m]) > 0 for m in ['40','100','200','400']) == 4)}")

        samples_per_epoch = train_stats['total_samples_per_epoch']
        # Dynamic batch size adjustment for MAXIMUM utilization Ensure batch size doesn't exceed reasonable limits for stability
        effective_batch_size = min(max(16, samples_per_epoch // 200), 32)
        print(f"Inner training samples: {samples_per_epoch}, batch size: {effective_batch_size}")
        
        sampler = train_ds.get_class_balanced_sampler()
        train_loader = DataLoader(
            train_ds, batch_size=effective_batch_size,
            sampler=sampler if sampler else None,
            shuffle=(sampler is None),
            num_workers=config['num_workers'], pin_memory=config['pin_memory'],
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_ds, batch_size=config['batch_size'], shuffle=False, 
            num_workers=config['num_workers'], pin_memory=config['pin_memory']
        )
        val_loader = DataLoader(
            val_ds, batch_size=config['batch_size'], shuffle=False, 
            num_workers=config['num_workers'], pin_memory=config['pin_memory']
        )

        train_labels = [train_ds.patient_dict[pid]['label'] for pid in train_pats]
        class_weights = calculate_class_weights(train_labels).to(device)
        print(f"Class weights: Benign={class_weights[0]:.2f}, Malignant={class_weights[1]:.2f}")
        
        epochs = NUM_EPOCHS
        model = SimpleConcatBaseline().to(device)
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, weight=class_weights, label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=LR_SCHEDULER_FACTOR, 
            patience=LR_SCHEDULER_PATIENCE
        )
        
        best_val_bal_acc = 0
        epochs_no_improve = 0
        best_model_state = None
        optimal_threshold = 0.5
        
        # Track metrics for learning curves and analysis
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        val_metrics_history = []
        overfitting_patience = 5
        overfitting_threshold = 0.1
        importance = {}
        
        for epoch in range(1, epochs+1):
            # Set epoch for deterministic sampling diversity
            train_ds.set_epoch(epoch)
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, 
                use_mixup=True,
                mixup_alpha=MIXUP_ALPHA
            )
            val_loss, val_acc, val_bal, val_f1, val_auc, val_prec, val_rec, threshold = eval_model_with_threshold_optimization(
                model, val_loader, criterion, device, mc_dropout=True
            )
            scheduler.step(val_bal) 
             
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            val_metrics_history.append({
                'epoch': epoch,
                'val_acc': val_acc,
                'val_bal': val_bal,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'val_prec': val_prec,
                'val_rec': val_rec
            })
            
            
            # Overfitting detection
            overfitting_warning = ""
            perfect_validation_warning = ""
            if val_loss > train_loss + 0.15:
                overfitting_warning = " [VAL LOSS DIVERGENCE]"
            
            # Check for train-validation loss divergence
            if len(train_losses) >= overfitting_patience:
                recent_train_loss = np.mean(train_losses[-overfitting_patience:])
                recent_val_loss = np.mean(val_losses[-overfitting_patience:])
                if recent_val_loss > recent_train_loss + overfitting_threshold:
                    overfitting_warning = " [TRAIN-VAL DIVERGENCE]"
            
            print(f"Epoch {epoch:02d}: "
                  f"Train: Loss {train_loss:.4f}, Acc {train_acc:.3f} | "
                  f"Val: Loss {val_loss:.4f}, Acc {val_acc:.3f}, "
                  f"BalAcc {val_bal:.3f}, F1 {val_f1:.3f}, AUC {val_auc:.3f}, "
                  f"Prec {val_prec:.3f}, Rec {val_rec:.3f}, Thresh {threshold:.3f} | LR: {optimizer.param_groups[0]['lr']:.6f} "
                  f"{overfitting_warning}{perfect_validation_warning}")
            
            if val_bal > best_val_bal_acc:
                best_val_bal_acc = val_bal
                best_model_state = model.state_dict().copy()
                optimal_threshold = threshold
                epochs_no_improve = 0
                print(f"✅ New best validation balanced accuracy: {best_val_bal_acc:.3f}, threshold: {optimal_threshold:.3f}")
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"⚠️ Early stopping after {epoch} epochs (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break
        
        # Load best model and evaluate on test set with optimized threshold
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"✅ Best model loaded (Val BalAcc: {best_val_bal_acc:.3f})")

        # Final test evaluation with optimized threshold (NO threshold optimization on test set)
        metrics = eval_model(
            model, test_loader, criterion, device, optimal_threshold
        )

        print(f"⚡️ Test Results: Acc {metrics['accuracy']:.3f}, BalAcc {metrics['balanced_accuracy']:.3f}, F1 {metrics['f1_score']:.3f}, AUC {metrics['auc']:.3f}, Precision {metrics['precision']:.3f}, Recall {metrics['recall']:.3f} (threshold: {optimal_threshold:.3f})")
        
        # Print confusion matrix
        print(f"📊 Confusion Matrix (Fold {fold_idx}):")
        print(f"   [[TN: {metrics['confusion_matrix'][0][0]:3d}, FP: {metrics['confusion_matrix'][0][1]:3d}]")
        print(f"    [FN: {metrics['confusion_matrix'][1][0]:3d}, TP: {metrics['confusion_matrix'][1][1]:3d}]]")
        print(f"⚡ Avg Inference Time: {metrics['avg_inference_time']:.4f}s per sample")

if __name__ == "__main__":
    main()