import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MultiScaleAttentionPool(nn.Module):
    def __init__(self, in_channels, scales=[1, 2, 4]):
        super().__init__()
        self.scale_attentions = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, in_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, 1, 1),
                nn.Sigmoid()
            ) for scale in scales
        ])
        self.scale_fusion = nn.Conv2d(len(scales), 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        scale_maps = []
        for module in self.scale_attentions:
            attn = module(x)
            attn = F.interpolate(attn, size=(H, W), mode='bilinear', align_corners=False)
            scale_maps.append(attn)
        stacked = torch.cat(scale_maps, dim=1)
        fused_attn = self.sigmoid(self.scale_fusion(stacked))
        attended = x * fused_attn
        pooled = F.adaptive_avg_pool2d(attended, 1).flatten(1)
        return pooled, fused_attn


class ClinicalChannelAttention(nn.Module):
    def __init__(self, channels, reduction=12):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.global_max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        avg = self.global_avg(x).view(b, c)
        mx = self.global_max(x).view(b, c)
        combined = torch.cat([avg, mx], dim=1)
        attn = self.fc(combined).view(b, c, 1, 1)
        return x * attn.expand_as(x), attn.squeeze()


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand.floor_()
        return x * rand / keep_prob


class MMNet(nn.Module):
    def __init__(self, magnifications=['40', '100', '200', '400'], num_classes=2, dropout=0.3, backbone='efficientnet_b1'):
        super().__init__()
        self.magnifications = magnifications
        self.extractors = nn.ModuleDict({
            f'extractor_{mag}x': timm.create_model(backbone, pretrained=True, num_classes=0, global_pool='', drop_rate=dropout * 0.5)
            for mag in magnifications
        })

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(next(self.parameters()).device)
            self.feat_channels = self.extractors['extractor_40x'](dummy).shape[1]

        self.spatial_att = nn.ModuleDict({
            f'sp_att_{mag}x': MultiScaleAttentionPool(self.feat_channels)
            for mag in magnifications
        })

        self.channel_att = nn.ModuleDict({
            f'ch_att_{mag}x': ClinicalChannelAttention(self.feat_channels)
            for mag in magnifications
        })

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_channels * len(magnifications), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(512, num_classes)
        )

    def forward(self, images_dict):
        features = []
        for mag in self.magnifications:
            x = images_dict[f'mag_{mag}']
            x = self.extractors[f'extractor_{mag}x'](x)
            x = StochasticDepth(0.1)(x)
            x, _ = self.channel_att[f'ch_att_{mag}x'](x)
            x, _ = self.spatial_att[f'sp_att_{mag}x'](x)
            features.append(x)

        fused = torch.cat(features, dim=1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits

    def get_magnification_importance(self):
        # Placeholder: add real mag importance logic if using attention
        return {mag: 1.0 / len(self.magnifications) for mag in self.magnifications}

    def print_model_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MMNet (feat_dim={self.feat_channels})")
        print(f"Total params: {total:,}, Trainable: {trainable:,}")
