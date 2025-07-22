import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MultiScaleAttentionPool(nn.Module):
    def __init__(self, in_channels, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        # Attention modules for each scale
        self.scale_attentions = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, in_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, 1, 1),
                nn.Sigmoid()
            ) for scale in scales
        ])
        # Fuse multi-scale attention maps
        self.scale_fusion = nn.Conv2d(len(scales), 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        scale_maps = []
        # Generate attention map at each scale
        for scale, module in zip(self.scales, self.scale_attentions):
            attn = module(x)  # [B, 1, scale, scale]
            if attn.shape[-2] != H:
                attn = F.interpolate(attn, size=(H, W), mode='bilinear', align_corners=False)
            scale_maps.append(attn)
        # Fuse and apply
        stacked = torch.cat(scale_maps, dim=1)  # [B, num_scales, H, W]
        fused_attn = self.sigmoid(self.scale_fusion(stacked))  # [B, 1, H, W]
        attended = x * fused_attn
        pooled = F.adaptive_avg_pool2d(attended, 1).flatten(1)  # [B, C]
        return pooled, fused_attn


class HierarchicalMagnificationAttention(nn.Module):
    """
    Hierarchical attention across magnifications: 40x->100x->200x->400x
    """
    def __init__(self, feat_dim, num_heads=8):
        super().__init__()
        self.feat_dim = feat_dim
        self.mag_hierarchy = ['40', '100', '200', '400']
        # Learnable embeddings per magnification
        self.mag_embeddings = nn.Parameter(torch.randn(len(self.mag_hierarchy), feat_dim))
        # Multi-head attention for hierarchy
        self.attention_layers = nn.ModuleDict({
            mag: nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
            for mag in self.mag_hierarchy[1:]
        })
        # LayerNorm for residual
        self.layer_norms = nn.ModuleDict({
            mag: nn.LayerNorm(feat_dim)
            for mag in self.mag_hierarchy[1:]
        })

    def forward(self, mag_features):
        # Add magnification embeddings
        enhanced = {}
        for i, mag in enumerate(self.mag_hierarchy):
            enhanced[mag] = mag_features[mag] + self.mag_embeddings[i]
        # 40x passes through
        hierarchical = {'40': enhanced['40']}
        attention_maps = {}
        # Apply hierarchical attention for each subsequent magnification
        for i, mag in enumerate(self.mag_hierarchy[1:], 1):
            contexts = self.mag_hierarchy[:i]
            context_feats = torch.stack([hierarchical[c] for c in contexts], dim=1)  # [B, i, feat_dim]
            query = enhanced[mag].unsqueeze(1)  # [B, 1, feat_dim]
            attended, weights = self.attention_layers[mag](query, context_feats, context_feats)
            # Residual + norm
            fused = self.layer_norms[mag](enhanced[mag] + attended.squeeze(1))
            hierarchical[mag] = fused
            attention_maps[mag] = weights  # [B, 1, i]
        return hierarchical, attention_maps


class ClinicalChannelAttention(nn.Module):
    """
    Enhanced channel attention with dual pooling for clinical features
    """
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
        # x: [B, C, H, W]
        b, c = x.shape[0], x.shape[1]
        avg = self.global_avg(x).view(b, c)
        mx = self.global_max(x).view(b, c)
        combined = torch.cat([avg, mx], dim=1)
        attn = self.fc(combined).view(b, c, 1, 1)
        return x * attn.expand_as(x), attn.squeeze()


class StochasticDepth(nn.Module):
    """
    Stochastic Depth regularization: randomly drops residual branches
    """
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x, skip=None):
        if not self.training or self.drop_prob == 0.0:
            return x + (skip if skip is not None else 0)
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand.floor_()
        if skip is not None:
            return skip + x * rand / keep_prob
        return x * rand / keep_prob


class ClinicalCrossMagFusion(nn.Module):
    """
    Fusion across magnifications with attention + residual concat
    """
    def __init__(self, feat_dim, num_mags):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_mags = num_mags
        self.mag_importance = nn.Parameter(torch.ones(num_mags))
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=8, batch_first=True, dropout=0.1
        )
        self.fusion_layers = nn.Sequential(
            nn.Linear(feat_dim * num_mags, feat_dim * 2),
            nn.BatchNorm1d(feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.BatchNorm1d(feat_dim),
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, feats_dict):
        # Stack and weight
        feats = list(feats_dict.values())  # [B, C]
        stacked = torch.stack(feats, dim=1)  # [B, num_mags, C]
        weights = F.softmax(self.mag_importance, dim=0)
        weighted = stacked * weights.view(1, -1, 1)
        # Cross-attention
        attn_out, attn_w = self.multihead_attn(weighted, weighted, weighted)
        global_feats = attn_out.mean(dim=1)  # [B, C]
        # Residual concat fusion
        concat_feats = torch.cat(feats, dim=1)  # [B, C * num_mags]
        fused_concat = self.fusion_layers(concat_feats)
        final = self.residual_weight * global_feats + (1 - self.residual_weight) * fused_concat
        return final, attn_w


class AttentionVisualization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spatial_maps, hier_maps, cross_maps, mag_importance):
        return {
            'spatial': spatial_maps,
            'hierarchical': hier_maps,
            'cross': cross_maps,
            'importance': mag_importance
        }


class MMNet(nn.Module):
    def __init__(
        self,
        magnifications=['40', '100', '200', '400'],
        num_classes=2,
        num_tumor_types=8,
        backbone='efficientnet_b1',
        dropout=0.3,
        stochastic_depth_prob=0.2
    ):
        super().__init__()
        self.magnifications = magnifications
        self.num_mags = len(magnifications)
        # Feature extractors
        self.extractors = nn.ModuleDict({
            f'extractor_{mag}x': timm.create_model(
                backbone, pretrained=True, num_classes=0, global_pool='', drop_rate=dropout * 0.5
            ) for mag in magnifications
        })
        # Infer feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            sample = self.extractors['extractor_40x'](dummy)
            self.feat_channels = sample.shape[1]

        # Attention modules
        self.spatial_attentions = nn.ModuleDict({
            f'spatial_pool_{mag}x': MultiScaleAttentionPool(self.feat_channels)
            for mag in magnifications
        })

        self.channel_attentions = nn.ModuleDict({
            f'channel_att_{mag}x': ClinicalChannelAttention(self.feat_channels)
            for mag in magnifications
        })

        self.stochastic_depths = nn.ModuleDict({
            f'sd_{mag}x': StochasticDepth(stochastic_depth_prob)
            for mag in magnifications
        })

        self.hierarchical_attention = HierarchicalMagnificationAttention(self.feat_channels)
        self.cross_mag_fusion = ClinicalCrossMagFusion(self.feat_channels, self.num_mags)
        
        # Classification heads
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self.tumor_classifier = nn.Sequential(
            nn.Linear(self.feat_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_tumor_types)
        )
        
        self.attention_viz = AttentionVisualization()

    def forward(self, images_dict, return_attention=False, return_features=False):
        # Step 1: extract + stochastic depth
        raw_feats = {}
        for mag in self.magnifications:
            x = images_dict[f'mag_{mag}']
            feat = self.extractors[f'extractor_{mag}x'](x)
            feat = self.stochastic_depths[f'sd_{mag}x'](feat)
            raw_feats[mag] = feat
        
        # Step 2: spatial + channel attention
        spatial_feats, spatial_maps = {}, {}
        channel_feats, channel_maps = {}, {}
        for mag in self.magnifications:
            sp, sp_map = self.spatial_attentions[f'spatial_pool_{mag}x'](raw_feats[mag])
            spatial_feats[mag], spatial_maps[mag] = sp, sp_map
            c_in = sp.unsqueeze(-1).unsqueeze(-1)
            ch, ch_map = self.channel_attentions[f'channel_att_{mag}x'](c_in)
            channel_feats[mag] = ch.squeeze(-1).squeeze(-1)
            channel_maps[mag] = ch_map
        
        # Step 3: hierarchical attention
        hier_feats, hier_maps = self.hierarchical_attention(channel_feats)
        
        # Step 4: cross-mag fusion
        fused, cross_maps = self.cross_mag_fusion(hier_feats)
        
        # Step 5: classification
        class_logits = self.classifier(fused)
        tumor_logits = self.tumor_classifier(fused)
        
        outputs = [class_logits, tumor_logits]
        if return_attention:
            outputs.append(self.attention_viz(spatial_maps, hier_maps, cross_maps, self.cross_mag_fusion.mag_importance))
        if return_features:
            outputs.append(fused)
        return tuple(outputs) if len(outputs) > 2 else (class_logits, tumor_logits)

    def get_attention_maps(self, images_dict):
        with torch.no_grad():
            _, _, att = self.forward(images_dict, return_attention=True)
            return att

    def get_magnification_importance(self):
        weights = F.softmax(self.cross_mag_fusion.mag_importance, dim=0)
        return {mag: float(weights[i]) for i, mag in enumerate(self.magnifications)}

    def get_confidence_scores(self, images_dict):
        with torch.no_grad():
            logits, _ = self.forward(images_dict)
            probs = F.softmax(logits, dim=1)
            return torch.max(probs, dim=1)[0]

    def print_model_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MMNet ({self.num_mags} mags, feat_dim={self.feat_channels})")
        print(f"Total params: {total:,}, Trainable: {trainable:,}")
