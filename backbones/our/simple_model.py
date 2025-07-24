import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SimpleMMNet(nn.Module):
    """Ultra-simple version of MMNet for debugging stability issues"""
    
    def __init__(self, magnifications=['40', '100', '200', '400'], num_classes=2, dropout=0.3):
        super().__init__()
        self.magnifications = magnifications
        
        # Single shared backbone - much simpler
        self.backbone = timm.create_model('resnet18', pretrained=True, num_classes=0, global_pool='avg')
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            self.feat_dim = self.backbone(dummy).shape[1]
        
        # Simple fusion - just average the features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes)
        )
        
        # Simple initialization
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, images_dict, mask=None):
        features = []
        
        for mag in self.magnifications:
            x = images_dict[f'mag_{mag}']
            feat = self.backbone(x)  # [B, feat_dim]
            features.append(feat)
        
        # Simple average fusion
        fused = torch.stack(features, dim=1).mean(dim=1)  # [B, feat_dim]
        
        # Classification
        logits = self.classifier(fused)
        return logits
    
    def get_magnification_importance(self, dataloader=None, device="cuda"):
        # Return equal weights for simplicity
        return {mag: 0.25 for mag in self.magnifications}