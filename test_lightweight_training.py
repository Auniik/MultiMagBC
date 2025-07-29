#!/usr/bin/env python3
"""
Quick test script to verify lightweight model training works
"""

import torch
from backbones.our.lightweight import MultiMagLightweightCNN
from config import get_training_config, FocalLoss, calculate_class_weights

def test_training_setup():
    print("🔧 Testing lightweight model training setup...")
    
    config = get_training_config()
    device = config['device']
    print(f"📱 Device: {device}")
    
    # Create model
    model = MultiMagLightweightCNN(num_classes=2, dropout=0.4).to(device)
    print(f"🧠 Model created with {model.get_model_info()['total_parameters']:,} parameters")
    
    # Test loss function
    class_weights = torch.tensor([1.5, 0.7]).to(device)  # Example weights
    criterion = FocalLoss(alpha=0.25, gamma=4.0, weight=class_weights)
    print(f"📐 Focal loss created with alpha=0.25, gamma=4.0")
    
    # Test forward and backward pass
    batch_size = 4
    test_input = {
        'mag_40': torch.randn(batch_size, 3, 224, 224).to(device),
        'mag_100': torch.randn(batch_size, 3, 224, 224).to(device),
        'mag_200': torch.randn(batch_size, 3, 224, 224).to(device),
        'mag_400': torch.randn(batch_size, 3, 224, 224).to(device)
    }
    test_labels = torch.randint(0, 2, (batch_size,)).to(device)
    
    # Forward pass
    model.train()
    outputs = model(test_input)
    print(f"✅ Forward pass: {outputs.shape}")
    
    # Loss calculation
    loss = criterion(outputs, test_labels)
    print(f"✅ Loss calculation: {loss.item():.4f}")
    
    # Backward pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=3e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"✅ Backward pass completed")
    
    # Test evaluation mode
    model.eval()
    with torch.no_grad():
        eval_outputs = model(test_input)
        print(f"✅ Evaluation mode: {eval_outputs.shape}")
    
    print(f"\n🎉 All training components work correctly!")
    print(f"💡 Ready to run full training with: python main.py")
    
    return True

if __name__ == "__main__":
    test_training_setup()