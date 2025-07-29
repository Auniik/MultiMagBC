#!/usr/bin/env python3
"""
Test TTA fix to ensure it returns all required metrics
"""

import torch
import numpy as np
from backbones.our.lightweight import MultiMagLightweightCNN
from utils.tta import evaluate_with_tta
from torch.utils.data import DataLoader, TensorDataset

def test_tta_fix():
    print("🔧 Testing TTA fix for KeyError: 'fpr'...")
    
    # Create lightweight model
    model = MultiMagLightweightCNN(num_classes=2)
    model.eval()
    
    # Create dummy test data
    batch_size = 8
    num_samples = 32
    
    # Create dummy images
    images_40 = torch.randn(num_samples, 3, 224, 224)
    images_100 = torch.randn(num_samples, 3, 224, 224)
    images_200 = torch.randn(num_samples, 3, 224, 224) 
    images_400 = torch.randn(num_samples, 3, 224, 224)
    
    # Create dummy masks and labels
    masks = torch.ones(num_samples, 4)  # All magnifications available
    labels = torch.randint(0, 2, (num_samples,))
    
    # Create dataset that returns the expected format
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            images_dict = {
                'mag_40': torch.randn(3, 224, 224),
                'mag_100': torch.randn(3, 224, 224),
                'mag_200': torch.randn(3, 224, 224),
                'mag_400': torch.randn(3, 224, 224)
            }
            mask = torch.ones(4)
            label = torch.randint(0, 2, (1,)).item() 
            return images_dict, mask, label
    
    dataset = DummyDataset(num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Test TTA evaluation
    try:
        metrics = evaluate_with_tta(model, dataloader, device='cpu', optimal_threshold=0.5)
        
        print("✅ TTA evaluation completed successfully!")
        print(f"✅ Returned metrics: {list(metrics.keys())}")
        
        # Check for required keys
        required_keys = ['accuracy', 'balanced_accuracy', 'f1_score', 'precision', 'recall', 'auc', 'confusion_matrix', 'avg_inference_time', 'fpr', 'tpr', 'thresholds']
        
        missing_keys = [key for key in required_keys if key not in metrics]
        if missing_keys:
            print(f"❌ Missing keys: {missing_keys}")
            return False
        else:
            print("✅ All required keys present!")
            
        # Check data types
        print(f"✅ Accuracy: {metrics['accuracy']:.3f}")
        print(f"✅ AUC: {metrics['auc']:.3f}")
        print(f"✅ FPR length: {len(metrics['fpr'])}")
        print(f"✅ TPR length: {len(metrics['tpr'])}")
        print(f"✅ Thresholds length: {len(metrics['thresholds'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ TTA evaluation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_tta_fix()
    if success:
        print(f"\n🎉 TTA fix successful! The KeyError should be resolved.")
    else:
        print(f"\n💥 TTA still has issues.")