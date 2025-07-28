import torch
import torch.nn as nn
import timm


class SimpleConcatBaseline(nn.Module):
    def __init__(self, backbone_name='mobilenetv3_small_100', num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feature_dim = self.backbone(dummy_input).shape[1]
        self.classifier = nn.Linear(feature_dim * 4, num_classes)  # 4 magnifications

    def forward(self, images_dict, mask=None):
        feats = [self.backbone(images_dict[f'mag_{m}']) for m in ['40', '100', '200', '400']]
        concat_feat = torch.cat(feats, dim=1)
        return self.classifier(concat_feat)
    

class BaselineSingleMag(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        return self.backbone(x)