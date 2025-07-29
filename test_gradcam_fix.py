#!/usr/bin/env python3
"""
Test script to verify GradCAM works with lightweight model
"""

import torch
from backbones.our.lightweight import MultiMagLightweightCNN
from evaluate.gradcam import GradCAM

def test_gradcam_lightweight():
    print("🔍 Testing GradCAM with lightweight model...")
    
    # Create lightweight model
    model = MultiMagLightweightCNN(num_classes=2)
    model.eval()
    
    # Create test input
    batch_size = 1
    test_input = {
        'mag_40': torch.randn(batch_size, 3, 224, 224),
        'mag_100': torch.randn(batch_size, 3, 224, 224),
        'mag_200': torch.randn(batch_size, 3, 224, 224),
        'mag_400': torch.randn(batch_size, 3, 224, 224)
    }
    
    # Test forward pass first
    with torch.no_grad():
        outputs = model(test_input)
        predicted = outputs.argmax(dim=1)
        print(f"✅ Model output shape: {outputs.shape}")
        print(f"✅ Predicted class: {predicted.item()}")
    
    # Test GradCAM
    try:
        gradcam = GradCAM(model)
        cams = gradcam.get_cam(test_input, target_class=predicted.item())
        
        print(f"✅ GradCAM completed successfully!")
        print(f"✅ Generated CAMs for magnifications: {list(cams.keys())}")
        
        # Check CAM shapes
        for mag, cam in cams.items():
            print(f"  - {mag}X CAM shape: {cam.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ GradCAM failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gradcam_lightweight()
    if success:
        print(f"\n🎉 GradCAM fix successful! Main pipeline should work now.")
    else:
        print(f"\n💥 GradCAM still has issues.")