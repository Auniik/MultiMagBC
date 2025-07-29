#!/usr/bin/env python3
"""
Test script for the lightweight model to verify integration
"""

import torch
from backbones.our.lightweight import MultiMagLightweightCNN

def test_lightweight_model():
    print("Testing MultiMagLightweightCNN integration...")
    
    # Create model
    model = MultiMagLightweightCNN(num_classes=2, dropout=0.4)
    
    # Print model info
    info = model.get_model_info()
    print("\n📊 Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value:,}")
    
    # Test forward pass
    batch_size = 4
    test_input = {
        'mag_40': torch.randn(batch_size, 3, 224, 224),
        'mag_100': torch.randn(batch_size, 3, 224, 224),
        'mag_200': torch.randn(batch_size, 3, 224, 224),
        'mag_400': torch.randn(batch_size, 3, 224, 224)
    }
    
    print(f"\n🧪 Testing forward pass with batch size {batch_size}...")
    
    # Normal forward
    with torch.no_grad():
        output = model(test_input)
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Output dtype: {output.dtype}")
    
    # Test magnification importance
    print(f"\n🎯 Testing magnification importance...")
    importance = model.get_magnification_importance()
    print(f"✅ Importance scores: {importance}")
    
    # Test attention maps
    print(f"\n🗺️ Testing attention maps...")
    attention_maps = model.get_attention_maps(test_input)
    print(f"✅ Fusion weights shape: {attention_maps['fusion_weights'].shape}")
    
    print(f"\n🎉 All tests passed! Lightweight model is ready.")
    return True

if __name__ == "__main__":
    test_lightweight_model()