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


class HierarchicalMagnificationAttention(nn.Module):
    def __init__(self, feat_dim, num_heads=8):
        super().__init__()
        self.mag_hierarchy = ['40', '100', '200', '400']
        self.embeddings = nn.Parameter(torch.randn(len(self.mag_hierarchy), feat_dim))
        
        # Shared cross-mag attention (bi-directional)
        self.attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feat_dim)

        # Learnable gating per magnification
        self.gates = nn.Parameter(torch.ones(len(self.mag_hierarchy)))

    def forward(self, mag_feats):
        # Add embeddings
        enhanced = []
        for i, mag in enumerate(self.mag_hierarchy):
            enhanced.append(mag_feats[mag] + self.embeddings[i])
        enhanced = torch.stack(enhanced, dim=1)  # [B, M, C]

        # Bi-directional attention across magnifications
        attn_out, _ = self.attn(enhanced, enhanced, enhanced)  # [B, M, C]

        # Gating between original + attended features
        gates = torch.sigmoid(self.gates).view(1, len(self.mag_hierarchy), 1)
        fused = self.norm(gates * attn_out + (1 - gates) * enhanced)

        # Return as dict
        return {mag: fused[:, i, :] for i, mag in enumerate(self.mag_hierarchy)}
    
class HybridCrossMagFusion(nn.Module):
    def __init__(self, channels_list, output_channels=256, num_heads=8, dropout=0.3):
        super().__init__()
        self.num_mags = len(channels_list)

        # Per-sample feature-driven importance (MLP-based)
        self.mag_importance_mlp = nn.Sequential(
            nn.Linear(output_channels, output_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(output_channels // 4, 1)
        )

        # Align features to common dimension
        self.align_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels, output_channels),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(inplace=True)
            ) for channels in channels_list
        ])

        # Cross-magnification attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_channels,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Fusion block
        self.fusion = nn.Sequential(
            nn.Linear(output_channels * self.num_mags, output_channels * 2),
            nn.BatchNorm1d(output_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_channels * 2, output_channels)
        )

        # Residual weighting
        self.res_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, *features_list, mask=None):
        # Step 1: Align features
        aligned_feats = []
        for features, align in zip(features_list, self.align_blocks):
            if features.ndim == 1:
                features = features.unsqueeze(0)
            aligned_feats.append(align(features))
        aligned_feats = torch.stack(aligned_feats, dim=1)  # [B, mags, C]

        # Step 2: Compute per-sample magnification weights
        B, M, C = aligned_feats.shape
        raw_weights = self.mag_importance_mlp(aligned_feats)  # [B, mags, 1]
        weights = F.softmax(raw_weights, dim=1)

        if mask is not None:
            mask = mask.unsqueeze(-1)  # [B, mags, 1]
            weights = weights * mask
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        weighted_feats = aligned_feats * weights  # [B, mags, C]

        # Step 3: Cross-magnification attention
        attn_out, _ = self.cross_attention(weighted_feats, weighted_feats, weighted_feats)
        global_feat = attn_out.mean(dim=1)  # [B, C]

        # Step 4: Concatenation-based fusion
        concat_feat = torch.cat([f for f in aligned_feats.unbind(dim=1)], dim=1)
        fused_feat = self.fusion(concat_feat)

        # Step 5: Residual combination
        return self.res_weight * global_feat + (1 - self.res_weight) * fused_feat


class MMNet(nn.Module):
    def __init__(self, magnifications=['40', '100', '200', '400'], num_classes=2, dropout=0.3, backbone='efficientnet_b1'):
        super().__init__()
        self.magnifications = magnifications
        self.extractors = nn.ModuleDict({
            f'extractor_{mag}x': timm.create_model(backbone, pretrained=True, num_classes=0, global_pool='', drop_rate=dropout * 0.5)
            for mag in magnifications
        })

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to('cpu')
            self.feat_channels = self.extractors['extractor_40x'](dummy).shape[1]

        self.spatial_att = nn.ModuleDict({
            f'sp_att_{mag}x': MultiScaleAttentionPool(self.feat_channels)
            for mag in magnifications
        })

        self.channel_att = nn.ModuleDict({
            f'ch_att_{mag}x': ClinicalChannelAttention(self.feat_channels)
            for mag in magnifications
        })

        self.hierarchical_attn = HierarchicalMagnificationAttention(self.feat_channels)
        self.cross_mag_fusion = HybridCrossMagFusion(
            channels_list=[self.feat_channels] * len(magnifications),
            output_channels=self.feat_channels,
            num_heads=8,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(512, num_classes)
        )
    
    def forward(self, images_dict, mask=None):
        # Extract features per magnification
        channel_outs = {}
        for mag in self.magnifications:
            x = images_dict[f'mag_{mag}']
            x = self.extractors[f'extractor_{mag}x'](x)
            x = StochasticDepth(0.1)(x)
            x, _ = self.channel_att[f'ch_att_{mag}x'](x)
            x, _ = self.spatial_att[f'sp_att_{mag}x'](x)
            channel_outs[mag] = x

        # Hierarchical magnification attention
        hier_feats = self.hierarchical_attn(channel_outs)
        
        # Convert dict → ordered list for fusion
        features_list = [hier_feats[mag] for mag in self.magnifications]

        # Cross-magnification fusion (now fully integrated)
        fused = self.cross_mag_fusion(*features_list, mask=mask)
        fused = self.dropout(fused)

        # Classification head
        logits = self.classifier(fused)
        return logits

    def print_model_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MMNet (feat_dim={self.feat_channels})")
        print(f"Total params: {total:,}, Trainable: {trainable:,}")

    def get_magnification_importance(self, dataloader=None, device="cuda"):
        self.eval()
        if dataloader is None:  # fallback: static equal weights
            with torch.no_grad():
                raw_weights = torch.ones(len(self.magnifications))
                return {mag: float((raw_weights / raw_weights.sum())[i].cpu()) for i, mag in enumerate(self.magnifications)}

        all_weights = []
        with torch.no_grad():
            for images_dict, mask, _ in dataloader:
                images = {k: v.to(device) for k, v in images_dict.items()}
                mask = mask.to(device)

                # Extract features for each magnification
                feats = []
                for i, mag in enumerate(self.magnifications):
                    x = self.extractors[f'extractor_{mag}x'](images[f'mag_{mag}'])
                    x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  # flatten
                    x = self.cross_mag_fusion.align_blocks[i](x)
                    feats.append(x)
                aligned_feats = torch.stack(feats, dim=1)  # [B, mags, C]

                # Get dynamic importance
                raw_weights = self.cross_mag_fusion.mag_importance_mlp(aligned_feats)  # [B, mags, 1]
                weights = F.softmax(raw_weights, dim=1)
                if mask is not None:
                    mask = mask.unsqueeze(-1)
                    weights = weights * mask
                    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

                all_weights.append(weights.squeeze(-1).cpu())

        mean_weights = torch.cat(all_weights, dim=0).mean(dim=0)
        return {mag: float(mean_weights[i]) for i, mag in enumerate(self.magnifications)}