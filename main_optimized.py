"""
OPTIMIZED MMNet - Fixed for 95% accuracy target
Main fixes:
1. Robust K-fold splitter that guarantees balanced folds
2. Optimized hyperparameters  
3. Stable training with ReduceLROnPlateau scheduler
4. Enhanced validation and early stopping
"""

import os
import json
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from backbones.our.lightweight import MultiMagLightweightCNN
from config_optimized import (SLIDES_PATH, LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE, 
                    LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR, DROPOUT_RATE, WEIGHT_DECAY,
                    SAMPLES_PER_PATIENT_BALANCED, EPOCH_MULTIPLIER_BALANCED, VAL_SAMPLES_PER_PATIENT_BALANCED,
                    FOCAL_ALPHA, FOCAL_GAMMA, LABEL_SMOOTHING, MIXUP_ALPHA, FocalLoss, 
                    get_training_config, calculate_class_weights, mixup_data, mixup_criterion)

from evaluate.gradcam import GradCAM, visualize_gradcam
from preprocess.robust_kfold_splitter import RobustPatientWiseKFoldSplitter
from torch.utils.data import DataLoader

from preprocess.multimagset import MultiMagPatientDataset
from preprocess.preprocess_light import get_fast_transforms as get_transforms, get_mixup_fn
from training.train_mm_k_fold import eval_model, eval_model_with_threshold_optimization, train_one_epoch
from sklearn.model_selection import train_test_split

from utils.stats import save_as_json
from utils.helpers import seed_everything
from utils.tta import evaluate_with_tta 

def boot(config):
    results_dir = os.path.join(config['output_dir'], 'results')
    csv_path = os.path.join(results_dir, 'results_summary.csv')
    csv_exists = os.path.exists(csv_path)
    if csv_exists:
        os.remove(csv_path)

def main():
    print("🚀 OPTIMIZED MMNet - Targeting 95% Accuracy with Robust Data Splitting")

    config = get_training_config()
    device = config['device']
    seed_everything(config['random_seed'])

    print(f"Using device: {device}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")

    boot(config)
    
    # USE ROBUST SPLITTER - PRESERVES NATURAL DATASET RATIO
    splitter = RobustPatientWiseKFoldSplitter(
        dataset_dir=SLIDES_PATH,
        n_splits=5,
        preserve_natural_ratio=True  # Handle natural class imbalance properly
    )
    splitter.print_summary()
    patient_dict = splitter.patient_dict

    train_transform, eval_transform, tta_transform = get_transforms()
    results_dir = os.path.join(config['output_dir'], 'results')

    fold_metrics = []
    importance_scores = []
    
    for fold_idx in range(len(splitter.folds)):
        print(f"\n===== Fold {fold_idx} =====")

        train_pats, val_pats, test_pats = splitter.get_fold(fold_idx)
        print(f"Train patients: {len(train_pats)}, Val Patients: {len(val_pats)}, Test patients: {len(test_pats)}")

        # OPTIMIZED DATASET PARAMETERS
        train_ds = MultiMagPatientDataset(
            patient_dict, train_pats, transform=train_transform, mode='train',
            sampling_mode='strict', class_balanced_sampling=True, subtype_balancing=True,
            samples_per_patient=SAMPLES_PER_PATIENT_BALANCED,
            epoch_multiplier=EPOCH_MULTIPLIER_BALANCED
        )
        val_ds = MultiMagPatientDataset(
            patient_dict, val_pats, transform=eval_transform, mode='val', 
            class_balanced_sampling=False, sampling_mode='strict',
            samples_per_patient=VAL_SAMPLES_PER_PATIENT_BALANCED
        )
        test_ds = MultiMagPatientDataset(
            patient_dict, test_pats, transform=eval_transform, mode='test',
            class_balanced_sampling=False, sampling_mode='relaxed'
        )

        train_stats = train_ds.get_sampling_stats()
        print(f"Training samples per epoch: {train_stats}")
        print(f"Validation samples: {len(val_ds)}, Test samples: {len(test_ds)}")

        samples_per_epoch = train_stats['total_samples_per_epoch']
        effective_batch_size = config['batch_size']
        print(f"Inner training samples: {samples_per_epoch}, batch size: {effective_batch_size}")
        
        sampler = train_ds.get_class_balanced_sampler()
        train_loader = DataLoader(
            train_ds, batch_size=effective_batch_size,
            sampler=sampler if sampler else None,
            shuffle=(sampler is None),
            num_workers=config['num_workers'], pin_memory=config['pin_memory'],
            persistent_workers=config.get('persistent_workers', False),
            prefetch_factor=config.get('prefetch_factor', 2),
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_ds, batch_size=config['batch_size'], shuffle=False, 
            num_workers=config['num_workers'], pin_memory=config['pin_memory'],
            persistent_workers=config.get('persistent_workers', False),
            prefetch_factor=config.get('prefetch_factor', 2)
        )
        val_loader = DataLoader(
            val_ds, batch_size=config['batch_size'], shuffle=False, 
            num_workers=config['num_workers'], pin_memory=config['pin_memory'],
            persistent_workers=config.get('persistent_workers', False),
            prefetch_factor=config.get('prefetch_factor', 2)
        )

        train_labels = [train_ds.patient_dict[pid]['label'] for pid in train_pats]
        
        epochs = NUM_EPOCHS
        model = MultiMagLightweightCNN(num_classes=2, dropout=DROPOUT_RATE).to(device)
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, weight=None, label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
        
        # USE STABLE REDUCE LR ON PLATEAU SCHEDULER
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, 
            patience=LR_SCHEDULER_PATIENCE
        )
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        optimal_threshold = 0.5
        best_val_bal_acc = 0 
        
        # Track metrics for analysis
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        val_metrics_history = []
        importance = {}
        
        for epoch in range(1, epochs+1):
            train_ds.set_epoch(epoch)
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, 
                scheduler=None,  # Don't pass scheduler to train_one_epoch
                use_mixup=True,
                mixup_alpha=MIXUP_ALPHA,
                epoch=epoch
            )
            
            history = eval_model_with_threshold_optimization(
                model, val_loader, criterion, device, mc_dropout=True
            )

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(history['val_loss'])
            val_accuracies.append(history['val_acc'])
            val_metrics_history.append({
                'epoch': epoch,
                'val_acc': history['val_acc'],
                'val_bal': history['val_bal'],
                'val_f1': history['val_f1'],
                'val_auc': history['val_auc'],
                'val_prec': history['val_prec'],
                'val_rec': history['val_rec']
            })
            
            # Step scheduler with validation loss
            scheduler.step(history['val_loss'])
            
            # Enhanced logging
            overfitting_warning = ""
            if history['val_loss'] > train_loss + 0.2:
                overfitting_warning = " [OVERFITTING WARNING]"
            
            print(f"Epoch {epoch:02d}: "
                  f"Train: Loss {train_loss:.4f}, Acc {train_acc:.3f} | "
                  f"Val: Loss {history['val_loss']:.4f}, Acc {history['val_acc']:.3f}, "
                  f"BalAcc {history['val_bal']:.3f}, F1 {history['val_f1']:.3f}, AUC {history['val_auc']:.3f}, "
                  f"Prec {history['val_prec']:.3f}, Rec {history['val_rec']:.3f}, Thresh {history['optimal_threshold']:.3f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}{overfitting_warning}")
            
            # Save best model based on validation loss
            if history['val_loss'] < best_val_loss:
                best_val_loss = history['val_loss']
                best_val_bal_acc = history['val_bal']
                best_model_state = model.state_dict().copy()
                optimal_threshold = history['optimal_threshold']
                epochs_no_improve = 0
                print(f" ✅ New best validation loss: {best_val_loss:.4f} (BalAcc: {best_val_bal_acc:.3f}, threshold: {optimal_threshold:.3f})")
                importance = model.get_magnification_importance(val_loader, device)
                print(f" 📊 Mag Importance: {importance}")
            else:
                epochs_no_improve += 1
            
            # Enhanced early stopping with minimum BalAcc requirement
            if best_val_bal_acc > 0.85 and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f" ✅ Early stopping after {epoch} epochs (BalAcc: {best_val_bal_acc:.3f} > 0.85)")
                break
            elif epochs_no_improve >= EARLY_STOPPING_PATIENCE * 2:
                print(f" ⚠️ Force early stopping after {epoch} epochs (no improvement for {EARLY_STOPPING_PATIENCE * 2} epochs)")
                break
        
        # Load best model and evaluate
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            ckpt_path = os.path.join(config['output_dir'], 'models', f"best_model_fold_{fold_idx}.pth")
            torch.save(best_model_state, ckpt_path)
            print(f"✅ Best model saved: {ckpt_path} (Val Loss: {best_val_loss:.4f}, BalAcc: {best_val_bal_acc:.3f})")
        
        # Test evaluation with TTA
        metrics = evaluate_with_tta(
            model, test_loader, device, optimal_threshold
        )

        print(f"⚡️ Test Results: Acc {metrics['accuracy']:.3f}, BalAcc {metrics['balanced_accuracy']:.3f}, "
              f"F1 {metrics['f1_score']:.3f}, AUC {metrics['auc']:.3f}, "
              f"Precision {metrics['precision']:.3f}, Recall {metrics['recall']:.3f} (threshold: {optimal_threshold:.3f})")
        
        # Detailed confusion matrix analysis
        cm = metrics['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"📊 Confusion Matrix (Fold {fold_idx}):")
        print(f"   [[TN: {tn:3d}, FP: {fp:3d}]  Specificity: {specificity:.3f}")
        print(f"    [FN: {fn:3d}, TP: {tp:3d}]]  Sensitivity: {sensitivity:.3f}")
        print(f"⚡ Avg Inference Time: {metrics['avg_inference_time']:.4f}s per sample")

        # Save detailed results
        fold_results = {
            'fold': fold_idx,
            'test_accuracy': metrics['accuracy'],
            'test_balanced_accuracy': metrics['balanced_accuracy'],
            'test_f1': metrics['f1_score'],
            'test_auc': metrics['auc'],
            'test_precision': metrics['precision'],
            'test_recall': metrics['recall'],
            'test_specificity': specificity,
            'test_sensitivity': sensitivity,
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
            'val_patients': len(val_pats),
            'test_patients': len(test_pats)
        }

        json_path = os.path.join(results_dir, f'fold_{fold_idx}_results.json')
        save_as_json(fold_results, json_path)
        
        # CSV summary
        csv_path = os.path.join(results_dir, 'results_summary.csv')
        csv_exists = os.path.exists(csv_path)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(['fold', 'accuracy', 'balanced_accuracy', 'f1', 'auc', 'precision', 'recall', 
                               'specificity', 'sensitivity', 'threshold', 'inference_time'])
            writer.writerow([fold_idx, metrics['accuracy'], metrics['balanced_accuracy'], metrics['f1_score'], 
                           metrics['auc'], metrics['precision'], metrics['recall'], specificity, sensitivity,
                           optimal_threshold, metrics['avg_inference_time']])

        importance = model.get_magnification_importance(test_loader, device)
        print(f"📌 Final Magnification Importance (Fold {fold_idx}): {importance}")
        print(f"💾 Results saved to: {json_path}")

        fold_metrics.append((metrics['accuracy'], metrics['balanced_accuracy'], metrics['f1_score'], 
                           metrics['auc'], metrics['precision'], metrics['recall']))

        # Generate GradCAM visualizations
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
    
    # COMPREHENSIVE RESULTS ANALYSIS
    accs, bals, f1s, aucs, precs, recs = zip(*fold_metrics)
    
    print("\n🎯 === OPTIMIZED CROSS-VALIDATION RESULTS ===")
    print(f"Accuracy:         {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"Balanced Acc:     {np.mean(bals):.3f} ± {np.std(bals):.3f}")
    print(f"F1 Score:         {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"AUC:              {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"Precision:        {np.mean(precs):.3f} ± {np.std(precs):.3f}")
    print(f"Recall:           {np.mean(recs):.3f} ± {np.std(recs):.3f}")
    
    # ACHIEVEMENT ANALYSIS
    target_bacc = 0.95
    folds_above_target = sum(1 for acc in bals if acc >= target_bacc)
    print(f"\n🏆 ACHIEVEMENT ANALYSIS:")
    print(f"Target BalAcc:    {target_bacc:.1%}")
    print(f"Folds ≥ target:   {folds_above_target}/{len(bals)} ({folds_above_target/len(bals):.1%})")
    print(f"Min BalAcc:       {min(bals):.3f}")
    print(f"Max BalAcc:       {max(bals):.3f}")
    print(f"BalAcc Range:     {max(bals) - min(bals):.3f}")
    
    if np.mean(bals) >= target_bacc:
        print("🎉 TARGET ACHIEVED! Average BalAcc ≥ 95%")
    elif min(bals) >= 0.90:
        print("✅ EXCELLENT! All folds ≥ 90% BalAcc")
    elif np.std(bals) < 0.05:
        print("📈 STABLE PERFORMANCE! Low variance achieved")
    else:
        print("⚠️ TARGET NOT MET - Check fold balance and hyperparameters")
    
    # Save comprehensive results
    cv_results_path = os.path.join(results_dir, 'cross_validation_results.json')
    cv_results = {
        'Accuracy': {'mean': np.mean(accs), 'std': np.std(accs), 'min': min(accs), 'max': max(accs)},
        'Balanced Accuracy': {'mean': np.mean(bals), 'std': np.std(bals), 'min': min(bals), 'max': max(bals)},
        'F1 Score': {'mean': np.mean(f1s), 'std': np.std(f1s), 'min': min(f1s), 'max': max(f1s)},
        'AUC': {'mean': np.mean(aucs), 'std': np.std(aucs), 'min': min(aucs), 'max': max(aucs)},
        'Precision': {'mean': np.mean(precs), 'std': np.std(precs), 'min': min(precs), 'max': max(precs)},
        'Recall': {'mean': np.mean(recs), 'std': np.std(recs), 'min': min(recs), 'max': max(recs)},
        'achievement': {
            'target_bacc': target_bacc,
            'folds_above_target': folds_above_target,
            'total_folds': len(bals),
            'achievement_rate': folds_above_target/len(bals),
            'target_achieved': np.mean(bals) >= target_bacc
        }
    }
    with open(cv_results_path, 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    print(f"\n💾 Comprehensive results saved to: {cv_results_path}")

if __name__ == "__main__":
    main()