import numpy as np
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


class BidirectionalMagnificationAttention(nn.Module):
    def __init__(self, feat_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.mag_hierarchy = ['40', '100', '200', '400']
        self.embeddings = nn.Parameter(torch.randn(len(self.mag_hierarchy), feat_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim)
        )
        self.norm2 = nn.LayerNorm(feat_dim)

        # For inspection
        self.last_attn_weights = None

    def forward(self, mag_feats):
        # Stack features into [B, 4, C] and add learnable embeddings
        feats = torch.stack([mag_feats[mag] for mag in self.mag_hierarchy], dim=1)  # [B, 4, C]
        feats = feats + self.embeddings.unsqueeze(0)  # add embeddings

        # Self-attention (all magnifications attend to each other)
        attn_out, attn_weights = self.attn(feats, feats, feats)  # [B, 4, C]
        self.last_attn_weights = attn_weights.detach().cpu().numpy()  # Save for analysis

        # Residual + normalization
        feats = self.norm(feats + attn_out)

        # Feed-forward network (Transformer block style)
        ffn_out = self.ffn(feats)
        feats = self.norm2(feats + ffn_out)

        # Return as dict for downstream layers
        return {mag: feats[:, i, :] for i, mag in enumerate(self.mag_hierarchy)}

    def get_last_attn_weights(self):
        """
        Returns attention weights as a JSON-friendly dictionary:
        {
            'magnifications': ['40', '100', '200', '400'],
            'weights': [[...], [...], [...], [...]]  # averaged attention matrix
        }
        """
        if self.last_attn_weights is None:
            return None

        # Average over batch & heads -> shape [query_len, key_len]
        attn_mean = self.last_attn_weights.mean(axis=(0, 1))  # numpy array [4, 4]

        # Convert to list for JSON
        return {
            "magnifications": self.mag_hierarchy,
            "weights": attn_mean.tolist()  # List of lists
        }
    
class CrossMagFusion(nn.Module):
    def __init__(self, channels_list, output_channels=256, dropout=0.3):
        super().__init__()
        self.num_mags = len(channels_list)

        self.temperature = nn.Parameter(torch.tensor(0.5))

        # Align features to a common dimension
        self.align_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ch, output_channels),
                nn.LayerNorm(output_channels),  # replaced
                nn.ReLU(inplace=True)
            ) for ch in channels_list
        ])

        # Magnification-level importance (gating)
        self.mag_importance_mlp = nn.Sequential(
            nn.Linear(output_channels, output_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(output_channels // 4, 1)  # scalar importance per mag
        )

        # Final fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(output_channels * self.num_mags, output_channels * 2),
            nn.LayerNorm(output_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_channels * 2, output_channels)
        )
    
    def forward(self, *features_list, mask=None):
        # Align features
        aligned_feats = []
        for feat, align in zip(features_list, self.align_blocks):
            if feat.ndim == 1:
                feat = feat.unsqueeze(0)
            aligned_feats.append(align(feat))
        aligned_feats = torch.stack(aligned_feats, dim=1)  # [B, mags, C]

        # Magnification weights
        raw_weights = self.mag_importance_mlp(aligned_feats)  # [B, mags, 1]

        weights = F.softmax(raw_weights / self.temperature.clamp(0.1, 1.0), dim=1)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            weights = weights * mask
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted features
        weighted_feats = aligned_feats * weights

        # Flatten & fuse
        concat_feat = weighted_feats.flatten(start_dim=1)
        fused_feat = self.fusion(concat_feat)

        return fused_feat, weights


class MMNet(nn.Module):
    def __init__(self, magnifications=['40', '100', '200', '400'], num_classes=2, dropout=0.3, backbone='efficientnet_b1'):
        super().__init__()
        self.magnifications = magnifications
        self.extractors = nn.ModuleDict({
            f'extractor_{mag}x': timm.create_model(
                backbone,
                pretrained=True,
                num_classes=0,
                global_pool='',
                drop_rate=dropout * 0.5,
                drop_path_rate=0.2 
            )
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

        self.hierarchical_attn = BidirectionalMagnificationAttention(self.feat_channels)
        self.cross_mag_fusion = CrossMagFusion(
            channels_list=[self.feat_channels] * len(magnifications),
            output_channels=self.feat_channels,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_channels, 512),
            nn.LayerNorm(512),  # replaced
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(512, num_classes)
        )
    
    def forward(self, images_dict, mask=None):
        channel_outs = {}
        for mag in self.magnifications:
            x = images_dict[f'mag_{mag}']
            x = self.extractors[f'extractor_{mag}x'](x)
            x, _ = self.channel_att[f'ch_att_{mag}x'](x)
            x, _ = self.spatial_att[f'sp_att_{mag}x'](x)
            channel_outs[mag] = x

        hier_feats = self.hierarchical_attn(channel_outs)
        features_list = [hier_feats[mag] for mag in self.magnifications]

        fused, weights = self.cross_mag_fusion(*features_list, mask=mask)  # now returns both
        fused = self.dropout(fused)

        logits = self.classifier(fused)
        return logits, weights

    def print_model_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MMNet (feat_dim={self.feat_channels})")
        print(f"Total params: {total:,}, Trainable: {trainable:,}")

    def get_magnification_importance(self, dataloader=None, device="cuda"):
        """
        Compute mean magnification importance across a dataset using the trained fusion MLP.
        Falls back to uniform weights if no dataloader is provided.
        """
        self.eval()

        # Fallback: no dataloader → return equal weights
        if dataloader is None:
            with torch.no_grad():
                raw_weights = torch.ones(len(self.magnifications))
                return {mag: float((raw_weights / raw_weights.sum())[i].cpu()) for i, mag in enumerate(self.magnifications)}

        all_weights = []
        with torch.no_grad():
            for images_dict, mask, _ in dataloader:
                images = {k: v.to(device) for k, v in images_dict.items()}
                mask = mask.to(device)

                # Stage 1: Extract features
                channel_outs = {}
                for mag in self.magnifications:
                    x = self.extractors[f'extractor_{mag}x'](images[f'mag_{mag}'])
                    x, _ = self.channel_att[f'ch_att_{mag}x'](x)
                    x, _ = self.spatial_att[f'sp_att_{mag}x'](x)
                    channel_outs[mag] = x

                # Stage 2: Hierarchical (bidirectional) attention
                hier_feats = self.hierarchical_attn(channel_outs)
                feats = [self.cross_mag_fusion.align_blocks[i](hier_feats[mag]) for i, mag in enumerate(self.magnifications)]
                aligned_feats = torch.stack(feats, dim=1)  # [B, mags, C]

                num_mags = self.cross_mag_fusion.num_mags
                # Stage 3: Compute per-sample fusion weights
                raw_weights = self.cross_mag_fusion.mag_importance_mlp(aligned_feats)  # [B, mags, 1]
                weights = F.softmax(raw_weights, dim=1)

                # Apply mask (normalize only over available magnifications)
                if mask is not None:
                    mask = mask.unsqueeze(-1)
                    weights = weights * mask
                    denom = weights.sum(dim=1, keepdim=True)
                    weights = torch.where(denom > 0, weights / (denom + 1e-8), torch.full_like(weights, 1.0 / num_mags))

                all_weights.append(weights.squeeze(-1).cpu())

        # Average across all batches
        mean_weights = torch.cat(all_weights, dim=0).mean(dim=0)
        return {mag: float(mean_weights[i]) for i, mag in enumerate(self.magnifications)}
    
    def aggregate_attention(self, dataloader, device):
        """
        Computes average hierarchical attention across all samples.
        Returns: dict with magnifications and mean attention weights.
        """
        self.eval()
        all_attns = []

        with torch.no_grad():
            for images_dict, mask, _ in dataloader:
                # Step 1: Extract features
                channel_outs = {}
                for mag in self.magnifications:
                    x = self.extractors[f'extractor_{mag}x'](images_dict[f'mag_{mag}'].to(device))
                    x, _ = self.channel_att[f'ch_att_{mag}x'](x)
                    x, _ = self.spatial_att[f'sp_att_{mag}x'](x)
                    channel_outs[mag] = x

                # Step 2: Prepare stacked features for attention
                feats = torch.stack([channel_outs[mag] for mag in self.magnifications], dim=1)
                feats = feats + self.hierarchical_attn.embeddings.unsqueeze(0)

                # Step 3: Run multi-head attention directly
                _, attn_weights = self.hierarchical_attn.attn(feats, feats, feats)  # [B, heads, 4, 4]
                all_attns.append(attn_weights.cpu())

        if not all_attns:
            return None

        all_attns = torch.cat(all_attns, dim=0)  # [N, heads, 4, 4]
        mean_attn = all_attns.mean(dim=(0, 1))  # [4, 4]

        return {
            "magnifications": self.magnifications,
            "mean_weights": mean_attn.tolist()
        }