"""
MMNet - Multi-Magnification Network for BreakHis Dataset
Main entry point for data analysis and training
"""

import os
import json
import csv
import time
from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from backbones.our.model import MMNet
from config import (SLIDES_PATH, LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE, 
                    LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR, DROPOUT_RATE, WEIGHT_DECAY,
                    SAMPLES_PER_PATIENT_BALANCED, EPOCH_MULTIPLIER_BALANCED, VAL_SAMPLES_PER_PATIENT_BALANCED,
                    FOCAL_ALPHA, FOCAL_GAMMA, LABEL_SMOOTHING, MIXUP_ALPHA, GRAD_CLIP_NORM, FocalLoss, 
                    get_training_config, calculate_class_weights, mixup_data, mixup_criterion)
from evaluate.gradcam import GradCAM, visualize_gradcam
from preprocess.kfold_splitter import PatientWiseKFoldSplitter

from torch.utils.data import DataLoader


from preprocess.multimagset import MultiMagPatientDataset
from preprocess.preprocess import get_transforms
from training.train_mm_k_fold import eval_model, eval_model_with_threshold_optimization, train_one_epoch
from sklearn.model_selection import train_test_split

from utils.stats import save_as_json

def boot(config):
    results_dir = os.path.join(config['output_dir'], 'results')

    csv_path = os.path.join(results_dir, 'results_summary.csv')
    csv_exists = os.path.exists(csv_path)
    if csv_exists:
        os.remove(csv_path)



def main():
    print("MMNet - Multi-Magnification Network for Breast Cancer Classification")

    from utils.helpers import seed_everything
    config = get_training_config()
    device = config['device']
    seed_everything(config['random_seed'])

    print(f"Using device: {device}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")

    boot(config)
    
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

    results_dir = os.path.join(config['output_dir'], 'results')

    fold_metrics = []
    importance_scores = []
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
        model = MMNet(dropout=DROPOUT_RATE).to(device)
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, weight=class_weights, label_smoothing=LABEL_SMOOTHING)
        # Use AdamW with improved parameters for stability
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY,
            eps=1e-8,  # Increased epsilon for numerical stability
            betas=(0.9, 0.999),  # Standard beta values
            amsgrad=False  # Keep False for stability
        )
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
        overfitting_patience = 3  # Reduced for earlier detection
        overfitting_threshold = 0.05  # More sensitive threshold
        validation_loss_window = []  # Track validation loss for stability
        importance = {}
        
        for epoch in range(1, epochs+1):
            # Set epoch for deterministic sampling diversity
            train_ds.set_epoch(epoch)
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, 
                use_mixup=True,
                mixup_alpha=MIXUP_ALPHA,
                accumulation_steps=2  # Use gradient accumulation for stability
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
            
            
            # Enhanced overfitting detection
            overfitting_warning = ""
            perfect_validation_warning = ""
            
            # Check for validation loss divergence (more sensitive)
            if val_loss > train_loss + 0.1:
                overfitting_warning = " [VAL LOSS DIVERGENCE]"
            
            # Track validation loss stability
            validation_loss_window.append(val_loss)
            if len(validation_loss_window) > 5:
                validation_loss_window.pop(0)
                
            # Check for train-validation loss divergence
            if len(train_losses) >= overfitting_patience:
                recent_train_loss = np.mean(train_losses[-overfitting_patience:])
                recent_val_loss = np.mean(val_losses[-overfitting_patience:])
                if recent_val_loss > recent_train_loss + overfitting_threshold:
                    overfitting_warning = " [TRAIN-VAL DIVERGENCE]"
                    
            # Check for validation loss instability
            if len(validation_loss_window) >= 5:
                val_loss_std = np.std(validation_loss_window)
                if val_loss_std > 0.05:  # High variance in validation loss
                    overfitting_warning += " [VAL INSTABILITY]"
            
            # Check for suspiciously perfect validation (potential overfitting)
            if val_bal >= 0.99 and val_auc >= 0.99:
                perfect_validation_warning = " [PERFECT VAL - CHECK OVERFITTING]"
            
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
                importance = model.get_magnification_importance(val_loader, device)
                print(f"📊 Mag Importance (Val BalAcc: {val_bal:.3f}): {importance}")
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"⚠️ Early stopping after {epoch} epochs (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break
        
        # Load best model and evaluate on test set with optimized threshold
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            ckpt_path = os.path.join(config['output_dir'], 'models', f"best_model_fold_{fold_idx}.pth")
            torch.save(best_model_state, ckpt_path)
            print(f"✅ Best model saved: {ckpt_path} (Val BalAcc: {best_val_bal_acc:.3f})")
        
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

        fold_results = {
            'fold': fold_idx,
            'test_accuracy': metrics['accuracy'],
            'test_balanced_accuracy': metrics['balanced_accuracy'],
            'test_f1': metrics['f1_score'],
            'test_auc': metrics['auc'],
            'test_precision': metrics['precision'],
            'test_recall': metrics['recall'],
            'optimal_threshold': optimal_threshold,
            'confusion_matrix': metrics['confusion_matrix'],
            'inference_time': metrics['avg_inference_time'],
            'roc_data': {
                'fpr': metrics['fpr'],
                'tpr': metrics['tpr'],
                'thresholds': metrics['thresholds']
            },
            'magnification_importance': importance,
            'training_history': {
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'val_metrics': val_metrics_history
            },
            'train_patients': len(train_pats),
            'test_patients': len(test_pats),
            'train_samples': len(train_ds),
            'test_samples': len(test_ds)
        }

        
        json_path = os.path.join(results_dir, f'fold_{fold_idx}_results.json')
        save_as_json(fold_results, json_path)
        
        csv_path = os.path.join(results_dir, 'results_summary.csv')
        csv_exists = os.path.exists(csv_path)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(['fold', 'accuracy', 'balanced_accuracy', 'f1', 'auc', 'precision', 'recall', 'threshold', 'inference_time', 'train_patients', 'test_patients'])
            writer.writerow([fold_idx, metrics['accuracy'], metrics['balanced_accuracy'], metrics['f1_score'], metrics['auc'], metrics['precision'], metrics['recall'], optimal_threshold, metrics['avg_inference_time'], len(train_pats), len(test_pats)])

        importance = model.get_magnification_importance(test_loader, device)
        print(f"📌 Final Magnification Importance (Fold {fold_idx}): {importance}")
        print(f"💾 Results saved to: {json_path}")

        fold_metrics.append((metrics['accuracy'], metrics['balanced_accuracy'], metrics['f1_score'], metrics['auc'], metrics['precision'], metrics['recall']))

        # Generate GradCAM visualizations for this fold
        print(f"\n📊 Generating GradCAM visualizations for fold {fold_idx}...")
        gradcam = GradCAM(model)
        model.eval()
        gradcam_dir = os.path.join(config['output_dir'], 'gradcam')
        
        gradcam_count = 0
        sample_idx = 0
        for images_dict, mask, labels in test_loader:
            batch_size = labels.size(0)
            for j in range(batch_size):
                if sample_idx >= 5:
                    break
                single_images = {k: v[j:j+1].to(device) for k, v in images_dict.items()}
                single_mask = mask[j:j+1].to(device)
                single_label = labels[j:j+1].to(device)
                with torch.no_grad():
                    outputs = model(single_images, single_mask)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    _, predicted = logits.max(1)
                
                cams = gradcam.get_cam(single_images, target_class=predicted.item())
                visualize_gradcam(
                    cams, 
                    single_images, 
                    true_label=single_label.item(),
                    pred_label=predicted.item(),
                    save_path=os.path.join(gradcam_dir, f'fold_{fold_idx}_sample_{sample_idx}.png'),
                    show=False
                )
                sample_idx += 1
                gradcam_count += 1
            if sample_idx >= 5:
                break
        
        print(f"✅ Generated {gradcam_count} GradCAM visualizations for fold {fold_idx}")

        importance_scores.append({
            'fold': fold_idx,
            'importance': importance,
            'optimal_threshold': optimal_threshold
        })
    
    accs, bals, f1s, aucs, precs, recs = zip(*[(acc, bal, f1, auc, prec, rec) for acc, bal, f1, auc, prec, rec in fold_metrics])
    print("\n=== Cross-Validation Results ===")
    print(f"Acc:      {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"BalAcc:   {np.mean(bals):.3f} ± {np.std(bals):.3f}")
    print(f"F1:       {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"AUC:      {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"Precision: {np.mean(precs):.3f} ± {np.std(precs):.3f}")
    print(f"Recall:    {np.mean(recs):.3f} ± {np.std(recs):.3f}")
    # save cross-validation results
    
    cv_results_path = os.path.join(results_dir, 'cross_validation_results.json')
    cv_results = {
        'Accuracy': {'mean': np.mean(accs), 'std': np.std(accs)},
        'Balanced Accuracy': {'mean': np.mean(bals), 'std': np.std(bals)},
        'F1 Score': {'mean': np.mean(f1s), 'std': np.std(f1s)},
        'AUC': {'mean': np.mean(aucs), 'std': np.std(aucs)},
        'Precision': {'mean': np.mean(precs), 'std': np.std(precs)},
        'Recall': {'mean': np.mean(recs), 'std': np.std(recs)}
    }
    with open(cv_results_path, 'w') as f:
        json.dump(cv_results, f, indent=2)

if __name__ == "__main__":
    main()