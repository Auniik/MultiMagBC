import timm
import torch.nn as nn

class BaselineSingleMag(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        return self.backbone(x)