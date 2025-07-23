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
                    FOCAL_ALPHA, FOCAL_GAMMA, LABEL_SMOOTHING, MIXUP_ALPHA, FocalLoss, 
                    get_training_config, calculate_class_weights, mixup_data, mixup_criterion)
from evaluate.gradcam import GradCAM, visualize_gradcam
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
    importance_scores = []
    for fold_idx, (train_pats, test_pats) in enumerate(splitter.folds):
        print(f"\n===== Fold {fold_idx} =====")
        print(f"Train patients: {len(train_pats)}, Test patients: {len(test_pats)}")
        # Datasets with BALANCED maximum utilization
        from config import SAMPLES_PER_PATIENT_BALANCED, EPOCH_MULTIPLIER_BALANCED
        train_ds = MultiMagPatientDataset(
            patient_dict, train_pats, transform=train_transform,
            samples_per_patient=SAMPLES_PER_PATIENT_BALANCED,  # Balanced 5 samples
            epoch_multiplier=EPOCH_MULTIPLIER_BALANCED,     # 3x diverse combinations
            adaptive_sampling=True  # Balanced utilization strategy
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
        
        # Create inner validation dataset with BALANCED sampling
        from config import VAL_SAMPLES_PER_PATIENT_BALANCED
        val_ds = MultiMagPatientDataset(
            patient_dict, val_pats, transform=eval_transform,
            samples_per_patient=VAL_SAMPLES_PER_PATIENT_BALANCED,  # Balanced 2 samples
            epoch_multiplier=1,     # Single epoch for consistent validation
            adaptive_sampling=False  # Consistent validation sampling
        )
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, 
                              num_workers=config['num_workers'], pin_memory=config['pin_memory'])
        
        # Update training loader with BALANCED utilization sampling
        train_ds_inner = MultiMagPatientDataset(
            patient_dict, train_pats_inner, transform=train_transform,
            samples_per_patient=SAMPLES_PER_PATIENT_BALANCED,  # Balanced 5 samples
            epoch_multiplier=EPOCH_MULTIPLIER_BALANCED,     # 3x diverse combinations
            adaptive_sampling=True  # Balanced utilization strategy
        )
        
        # Adjust batch size for increased data volume
        inner_train_stats = train_ds_inner.get_sampling_stats()
        samples_per_epoch = inner_train_stats['total_samples_per_epoch']
        
        # Dynamic batch size adjustment for MAXIMUM utilization
        target_batch_size = min(config['batch_size'] * 2, 32)  # Double batch size for larger datasets
        
        # Ensure batch size doesn't exceed reasonable limits for stability
        if samples_per_epoch > 4000:
            effective_batch_size = min(target_batch_size, 32)
        elif samples_per_epoch > 2000:
            effective_batch_size = min(target_batch_size, 24)
        else:
            effective_batch_size = min(target_batch_size, 16)
            
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
        
        # Track metrics for learning curves and analysis
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        val_metrics_history = []
        overfitting_patience = 5
        overfitting_threshold = 0.1
        
        for epoch in range(1, epochs+1):
            # Set epoch for deterministic sampling diversity
            train_ds_inner.set_epoch(epoch)
            train_loss, train_acc = train_one_epoch(
                model, train_loader_inner, criterion, optimizer, device, 
                use_mixup=True, mixup_alpha=MIXUP_ALPHA
            )
            val_loss, val_acc, val_bal, val_f1, val_auc, val_prec, val_rec, threshold = eval_model_with_threshold_optimization(
                model, val_loader, criterion, device, use_dropout=True
            )
            
            # Track metrics for learning curves
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
                importance = model.get_magnification_importance()
                print(f"📊 Mag Importance (Val BalAcc: {val_bal:.3f}): {importance}")
            else:
                epochs_no_improve += 1
            
            # Early stopping
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
        _, test_acc, test_bal, test_f1, test_auc, test_prec, test_rec, _ = eval_model(model, test_loader, criterion, device, optimal_threshold)

        print(f"⚡️ Test Results: Acc {test_acc:.3f}, BalAcc {test_bal:.3f}, F1 {test_f1:.3f}, AUC {test_auc:.3f}, Precision {test_prec:.3f}, Recall {test_rec:.3f} (threshold: {optimal_threshold:.3f})")
        
        # Generate confusion matrix for this fold
        from sklearn.metrics import confusion_matrix
        test_labels = [test_ds.patient_dict[pid]['label'] for pid in test_pats]
        test_preds = []
        test_probs = []
        
        # Track inference time
        inference_start = time.time()
        with torch.no_grad():
            for images_dict, labels in test_loader:
                images = {k: v.to(device) for k, v in images_dict.items()}
                outputs = model(images)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= optimal_threshold).astype(int)
                test_preds.extend(preds)
                test_probs.extend(probs)
        inference_time = time.time() - inference_start
        avg_inference_time = inference_time / len(test_ds)
        
        cm = confusion_matrix(test_labels, test_preds)
        print(f"📊 Confusion Matrix (Fold {fold_idx}):")
        print(f"   [[TN: {cm[0,0]:3d}, FP: {cm[0,1]:3d}]")
        print(f"    [FN: {cm[1,0]:3d}, TP: {cm[1,1]:3d}]]")
        print(f"⚡ Avg Inference Time: {avg_inference_time:.4f}s per sample")
        
        # Generate ROC curve data
        fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
        
        # Save fold results
        fold_results = {
            'fold': fold_idx,
            'test_accuracy': test_acc,
            'test_balanced_accuracy': test_bal,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'optimal_threshold': optimal_threshold,
            'confusion_matrix': cm.tolist(),
            'inference_time': avg_inference_time,
            'roc_data': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
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
            'train_samples': len(train_ds_inner),
            'test_samples': len(test_ds)
        }
        
        # Save to JSON
        results_dir = os.path.join(config['output_dir'], 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        json_path = os.path.join(results_dir, f'fold_{fold_idx}_results.json')
        with open(json_path, 'w') as f:
            json.dump(fold_results, f, indent=2)
        
        # Save to CSV summary
        csv_path = os.path.join(results_dir, 'results_summary.csv')
        csv_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(['fold', 'accuracy', 'balanced_accuracy', 'f1', 'auc', 'precision', 'recall', 'threshold', 'inference_time', 'train_patients', 'test_patients'])
            writer.writerow([fold_idx, test_acc, test_bal, test_f1, test_auc, test_prec, test_rec, optimal_threshold, avg_inference_time, len(train_pats), len(test_pats)])
        
        importance = model.get_magnification_importance()
        print(f"📌 Final Magnification Importance (Fold {fold_idx}): {importance}")
        print(f"💾 Results saved to: {json_path}")
        
        fold_metrics.append((test_acc, test_bal, test_f1, test_auc, test_prec, test_rec))

        # Generate GradCAM visualizations for this fold
        print(f"\n📊 Generating GradCAM visualizations for fold {fold_idx}...")
        gradcam = GradCAM(model)
        model.eval()
        
        # Create gradcam output directory
        gradcam_dir = os.path.join(config['output_dir'], 'gradcam')
        os.makedirs(gradcam_dir, exist_ok=True)
        
        # Generate for first 3 test samples
        gradcam_count = 0
        for i, (images_dict, labels) in enumerate(test_loader):
            if i >= 3: 
                break
                
            images = {k: v.to(device) for k, v in images_dict.items()}
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(images)
                if isinstance(outputs, tuple):
                    logits = outputs[0]  # Binary classification output
                else:
                    logits = outputs
                _, predicted = logits.max(1)
            
            cams = gradcam.get_cam(images, target_class=predicted.item())
            save_path = os.path.join(gradcam_dir, f'fold_{fold_idx}_sample_{i}.png')
            visualize_gradcam(
                cams, 
                images, 
                true_label=labels.item(),
                pred_label=predicted.item(),
                save_path=save_path,
                show=False
            )
            gradcam_count += 1
        
        print(f"✅ Generated {gradcam_count} GradCAM visualizations for fold {fold_idx}")

        fold_metrics.append((test_acc, test_bal, test_f1, test_auc))
        importance_scores.append({
            'fold': fold_idx,
            'importance': importance,
            'optimal_threshold': optimal_threshold
        })
    
# Summary with all metrics
    accs, bals, f1s, aucs, precs, recs = zip(*[(acc, bal, f1, auc, prec, rec) for acc, bal, f1, auc, prec, rec in fold_metrics])
    print("\n=== Cross-Validation Results ===")
    print(f"Acc:      {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"BalAcc:   {np.mean(bals):.3f} ± {np.std(bals):.3f}")
    print(f"F1:       {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"AUC:      {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"Precision: {np.mean(precs):.3f} ± {np.std(precs):.3f}")
    print(f"Recall:    {np.mean(recs):.3f} ± {np.std(recs):.3f}")
    



if __name__ == "__main__":
    main()