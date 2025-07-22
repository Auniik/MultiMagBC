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
from config import SLIDES_PATH, LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE, LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR, get_training_config, calculate_class_weights
from preprocess.kfold_splitter import PatientWiseKFoldSplitter
import torchvision.transforms as T
from torch.utils.data import DataLoader


from preprocess.multimagset import MultiMagPatientDataset
from training.train_mm_k_fold import eval_model, train_one_epoch



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

    # Define transforms
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(0.1, 0.1, 0.1, 0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
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
        # Datasets & loaders
        train_ds = MultiMagPatientDataset(patient_dict, train_pats, transform=train_transform)
        test_ds = MultiMagPatientDataset(patient_dict, test_pats, transform=eval_transform)
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=config['pin_memory'], drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=config['pin_memory'])

        # Calculate class weights for imbalanced dataset
        train_labels = [train_ds.patient_dict[pid]['label'] for pid in train_pats]
        class_weights = calculate_class_weights(train_labels).to(device)
        print(f"Class weights: Benign={class_weights[0]:.2f}, Malignant={class_weights[1]:.2f}")
        
        # Model, criterion, optimizer, scheduler
        epochs = NUM_EPOCHS

        model = MMNet().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=LR_SCHEDULER_FACTOR, 
            patience=LR_SCHEDULER_PATIENCE
        )

        best_bal_acc = 0
        epochs_no_improve = 0
        best_model_state = None
        
        for epoch in range(1, epochs+1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_bal, val_f1, val_auc = eval_model(model, test_loader, criterion, device)
            
            # Step scheduler with validation balanced accuracy
            scheduler.step(val_bal)
            
            print(f"Epoch {epoch:02d}: "
                  f"Train: Loss {train_loss:.4f}, Acc {train_acc:.3f} |"
                  f"Val: Loss {val_loss:.4f}, Acc {val_acc:.3f}, "
                  f"BalAcc {val_bal:.3f}, F1 {val_f1:.3f}, AUC {val_auc:.3f}")
            
            # Save best model and implement early stopping
            if val_bal > best_bal_acc:
                best_bal_acc = val_bal
                best_model_state = model.state_dict().copy()
                epochs_no_improve = 0
                print(f"New best balanced accuracy: {best_bal_acc:.3f}")
            else:
                epochs_no_improve += 1
            
            # Early stopping
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {epoch} epochs (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break
        
        # Save best model for this fold
        if best_model_state is not None:
            ckpt_path = os.path.join(config['output_dir'], 'models', f"best_model_fold_{fold_idx}.pth")
            torch.save(best_model_state, ckpt_path)
            print(f"Best model saved: {ckpt_path} (BalAcc: {best_bal_acc:.3f})")

        # Use best metrics from this fold
        _, final_val_acc, final_val_bal, final_val_f1, final_val_auc = eval_model(model, test_loader, criterion, device)
        fold_metrics.append((final_val_acc, final_val_bal, final_val_f1, final_val_auc))
    # Summary
    accs, bals, f1s, aucs = zip(*fold_metrics)
    print("\n=== Cross-Validation Results ===")
    print(f"Acc: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"BalAcc: {np.mean(bals):.3f} ± {np.std(bals):.3f}")
    print(f"F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")



if __name__ == "__main__":
    main()