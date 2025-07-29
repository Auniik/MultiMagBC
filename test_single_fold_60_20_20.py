#!/usr/bin/env python3
"""
Test single fold training with 60-20-20 split to verify improved stability
"""

import torch
import torch.optim as optim
from preprocess.kfold_splitter import PatientWiseKFoldSplitter
from preprocess.multimagset import MultiMagPatientDataset
from preprocess.preprocess import get_transforms
from backbones.our.lightweight import MultiMagLightweightCNN
from config import SLIDES_PATH, get_training_config, FocalLoss, calculate_class_weights
from training.train_mm_k_fold import train_one_epoch, eval_model_with_threshold_optimization
from torch.utils.data import DataLoader

def test_single_fold_60_20_20():
    print("🔬 Testing single fold training with 60-20-20 split...")
    
    config = get_training_config()
    device = config['device']
    
    # Create splitter with 60-20-20 split
    splitter = PatientWiseKFoldSplitter(
        dataset_dir=SLIDES_PATH,
        n_splits=5,
        validation_split=0.25  # 60-20-20 split
    )
    
    # Test fold 0
    train_pats, val_pats, test_pats = splitter.get_fold(0)
    print(f"📊 Split sizes: Train={len(train_pats)}, Val={len(val_pats)}, Test={len(test_pats)}")
    print(f"📊 Split ratios: Train={len(train_pats)/82*100:.1f}%, Val={len(val_pats)/82*100:.1f}%, Test={len(test_pats)/82*100:.1f}%")
    
    patient_dict = splitter.patient_dict
    train_transform, eval_transform, _ = get_transforms()
    
    # Create datasets
    train_ds = MultiMagPatientDataset(patient_dict, train_pats, transform=train_transform, mode='train')
    val_ds = MultiMagPatientDataset(patient_dict, val_pats, transform=eval_transform, mode='val', full_utilization_mode='all')
    
    # Create data loaders
    sampler = train_ds.get_class_balanced_sampler()
    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    
    # Setup model and training
    model = MultiMagLightweightCNN(num_classes=2, dropout=0.4).to(device)
    
    train_labels = [train_ds.patient_dict[pid]['label'] for pid in train_pats]
    class_weights = calculate_class_weights(train_labels, method='balanced').to(device)
    
    criterion = FocalLoss(alpha=0.5, gamma=2.0, weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=3e-3)
    
    print(f"\n🏃‍♂️ Training 3 epochs to test stability...")
    
    for epoch in range(1, 4):
        train_ds.set_epoch(epoch)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=True, mixup_alpha=0.2
        )
        
        val_loss, val_acc, val_bal, val_f1, val_auc, val_prec, val_rec, threshold = eval_model_with_threshold_optimization(
            model, val_loader, criterion, device, mc_dropout=True
        )
        
        print(f"Epoch {epoch}: Train: Loss {train_loss:.4f}, Acc {train_acc:.3f} | "
              f"Val: Loss {val_loss:.4f}, Acc {val_acc:.3f}, BalAcc {val_bal:.3f}, "
              f"F1 {val_f1:.3f}, Thresh {threshold:.3f}")
        
        # Check magnification importance
        importance = model.get_magnification_importance(val_loader, device)
        print(f"  📊 Mag Importance: {importance}")
    
    print(f"\n✅ 60-20-20 split training completed successfully!")
    print(f"💡 Observations:")
    print(f"   • Validation set has {len(val_ds)} samples (vs ~241 before)")
    print(f"   • {len(val_pats)} validation patients (vs ~13 before)")
    print(f"   • Should provide more stable validation metrics")
    print(f"   • Ready for full 5-fold cross-validation")
    
    return True

if __name__ == "__main__":
    test_single_fold_60_20_20()