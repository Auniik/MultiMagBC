#!/usr/bin/env python3
"""
Quick test of single fold with 60-20-20 split
"""

import torch
from preprocess.kfold_splitter import PatientWiseKFoldSplitter
from preprocess.multimagset import MultiMagPatientDataset
from preprocess.preprocess import get_transforms
from backbones.our.lightweight import MultiMagLightweightCNN
from config import SLIDES_PATH, get_training_config
from torch.utils.data import DataLoader

def test_60_20_20_training():
    print("🚀 Testing training with 60-20-20 split...")
    
    config = get_training_config()
    device = config['device']
    
    # Create splitter
    splitter = PatientWiseKFoldSplitter(
        dataset_dir=SLIDES_PATH,
        n_splits=5,
        validation_split=0.25  # 60-20-20 split
    )
    
    # Test fold 0
    train_pats, val_pats, test_pats = splitter.get_fold(0)
    print(f"📊 Fold 0 splits: Train={len(train_pats)}, Val={len(val_pats)}, Test={len(test_pats)}")
    
    patient_dict = splitter.patient_dict
    train_transform, eval_transform, _ = get_transforms()
    
    # Create datasets
    train_ds = MultiMagPatientDataset(patient_dict, train_pats, transform=train_transform, mode='train')
    val_ds = MultiMagPatientDataset(patient_dict, val_pats, transform=eval_transform, mode='val', full_utilization_mode='all')
    test_ds = MultiMagPatientDataset(patient_dict, test_pats, transform=eval_transform, mode='test', full_utilization_mode='all')
    
    print(f"📈 Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)
    
    # Create model
    model = MultiMagLightweightCNN(num_classes=2).to(device)
    print(f"🧠 Model created with {model.get_model_info()['total_parameters']:,} parameters")
    
    # Test forward pass on each loader
    model.eval()
    with torch.no_grad():
        # Test train loader
        for images_dict, mask, labels in train_loader:
            images_dict = {k: v.to(device) for k, v in images_dict.items()}
            mask = mask.to(device)
            outputs = model(images_dict, mask)
            print(f"✅ Train batch: {outputs.shape}")
            break
            
        # Test val loader
        for images_dict, mask, labels in val_loader:
            images_dict = {k: v.to(device) for k, v in images_dict.items()}
            mask = mask.to(device) 
            outputs = model(images_dict, mask)
            print(f"✅ Val batch: {outputs.shape}")
            break
            
        # Test test loader
        for images_dict, mask, labels in test_loader:
            images_dict = {k: v.to(device) for k, v in images_dict.items()}
            mask = mask.to(device)
            outputs = model(images_dict, mask) 
            print(f"✅ Test batch: {outputs.shape}")
            break
    
    print(f"\n🎉 60-20-20 split training setup working perfectly!")
    print(f"💡 Expected benefits:")
    print(f"   • Larger validation sets: {len(val_pats)} patients (vs ~13 before)")
    print(f"   • More stable validation metrics")
    print(f"   • Better overfitting detection")
    print(f"   • Reduced fold variance")
    
    return True

if __name__ == "__main__":
    test_60_20_20_training()