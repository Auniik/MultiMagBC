"""
Quick script to analyze why we're getting large gradients in the simple model
"""

import torch
import torch.nn as nn
import timm

# Create the simple model
class SimpleMMNet(nn.Module):
    def __init__(self, magnifications=['40', '100', '200', '400'], num_classes=2, dropout=0.3):
        super().__init__()
        self.magnifications = magnifications
        
        # Single shared backbone
        self.backbone = timm.create_model('resnet18', pretrained=True, num_classes=0, global_pool='avg')
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            self.feat_dim = self.backbone(dummy).shape[1]
        
        # Simple fusion - average the features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes)
        )
        
        print(f"Model feature dim: {self.feat_dim}")
        
        # Check initial weights
        for name, param in self.named_parameters():
            if 'classifier' in name:
                print(f"{name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}, norm={param.norm().item():.6f}")
    
    def forward(self, images_dict, mask=None):
        features = []
        
        for mag in self.magnifications:
            x = images_dict[f'mag_{mag}']
            feat = self.backbone(x)  # [B, feat_dim]
            features.append(feat)
        
        # Simple average fusion
        fused = torch.stack(features, dim=1).mean(dim=1)  # [B, feat_dim]
        
        # Check feature statistics
        print(f"Fused features - mean: {fused.mean().item():.4f}, std: {fused.std().item():.4f}, "
              f"min: {fused.min().item():.4f}, max: {fused.max().item():.4f}")
        
        # Classification
        logits = self.classifier(fused)
        
        print(f"Logits - mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}, "
              f"min: {logits.min().item():.4f}, max: {logits.max().item():.4f}")
        
        return logits

def analyze_gradient_explosion():
    print("🔍 Analyzing potential gradient explosion causes...")
    
    # Create model
    model = SimpleMMNet()
    
    # Create dummy data
    batch_size = 16
    images_dict = {
        f'mag_{mag}': torch.randn(batch_size, 3, 224, 224) 
        for mag in ['40', '100', '200', '400']
    }
    labels = torch.randint(0, 2, (batch_size,))
    
    # Check what happens during forward/backward
    print("\n📊 Forward pass analysis:")
    logits = model(images_dict)
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    print("\n🔙 Backward pass analysis:")
    loss.backward()
    
    # Check gradients for each parameter
    total_grad_norm = 0
    max_grad_norm = 0
    problematic_params = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            max_grad_norm = max(max_grad_norm, grad_norm)
            
            print(f"{name}:")
            print(f"  Param norm: {param.norm().item():.6f}")
            print(f"  Grad norm: {grad_norm:.6f}")
            print(f"  Grad mean: {param.grad.mean().item():.6f}")
            print(f"  Grad std: {param.grad.std().item():.6f}")
            
            if grad_norm > 1.0:
                problematic_params.append((name, grad_norm))
    
    total_grad_norm = total_grad_norm ** 0.5
    print(f"\n📈 Overall gradient statistics:")
    print(f"Total gradient norm: {total_grad_norm:.6f}")
    print(f"Max individual gradient norm: {max_grad_norm:.6f}")
    
    if problematic_params:
        print(f"\n⚠️ Parameters with large gradients:")
        for name, grad_norm in problematic_params:
            print(f"  {name}: {grad_norm:.6f}")
    
    # Test with different learning rates
    print(f"\n🎯 Testing learning rate sensitivity:")
    for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        # Fresh forward/backward
        model.zero_grad()
        logits = model(images_dict)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Check if step would be stable
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        print(f"  LR {lr}: Gradient norm = {total_norm:.6f}")
        
        model.zero_grad()

if __name__ == "__main__":
    analyze_gradient_explosion()