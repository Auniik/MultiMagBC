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
from config import SLIDES_PATH, LEARNING_RATE, get_training_config
from preprocess.kfold_splitter import PatientWiseKFoldSplitter
import torchvision.transforms as T
from torch.utils.data import DataLoader


from preprocess.multimagset import MultiMagPatientDataset
from training.train_mm_k_fold import eval_model, train_one_epoch



def main():
    print("MMNet - Multi-Magnification Network for Breast Cancer Classification")

    from utils.helpers import seed_everything
    config = get_training_config()
    device = torch.device('cpu')
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
        # Datasets & loaders
        train_ds = MultiMagPatientDataset(patient_dict, train_pats, transform=train_transform)
        test_ds = MultiMagPatientDataset(patient_dict, test_pats, transform=eval_transform)
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=config['pin_memory'], drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=config['pin_memory'])

        # Model, criterion, optimizer, scheduler
        epochs = 5

        model = MMNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_bal_acc = 0
        for epoch in range(1, epochs+1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_bal, val_f1, val_auc = eval_model(model, test_loader, criterion, device)
            scheduler.step()
            print(f"Epoch {epoch:02d}:"
                  f"Train: Loss {train_loss:.4f}, Acc {train_acc:.3f} | "
                  f"Val: Loss {val_loss:.4f}, Acc {val_acc:.3f}, BalAcc {val_bal:.3f}, F1 {val_f1:.3f}, AUC {val_auc:.3f}")
            # Save best
            if val_bal > best_bal_acc:
                best_bal_acc = val_bal
                ckpt_path = os.path.join(config['output_dir'], f"fold_{fold_idx}_best.pth")
                torch.save(model.state_dict(), ckpt_path)

        fold_metrics.append((val_acc, val_bal, val_f1, val_auc))
    # Summary
    accs, bals, f1s, aucs = zip(*fold_metrics)
    print("\n=== Cross-Validation Results ===")
    print(f"Acc: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"BalAcc: {np.mean(bals):.3f} ± {np.std(bals):.3f}")
    print(f"F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")



if __name__ == "__main__":
    main()