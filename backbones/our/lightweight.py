#!/usr/bin/env python3
"""
MultiMagLightweightCNN - Lightweight Multi-Magnification Network for Histopathology

A parameter-efficient CNN designed for the BreakHis dataset that processes
multiple magnifications (40x, 100x, 200x, 400x) with attention mechanisms.

Key Features:
- Depthwise separable convolutions (MobileNet-inspired)
- Shared shallow features across magnifications
- Magnitude-specific processing branches
- Attention-guided feature selection
- Cross-magnification fusion
- ~200-500K parameters (vs 5M+ for EfficientNet-based models)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


class InvertedResidual(nn.Module):
    """
    MobileNetV2-style inverted residual block with depthwise separable convolution.
    
    This is the core building block that provides efficiency through:
    1. Expansion layer (1x1 conv)
    2. Depthwise convolution (3x3)
    3. Projection layer (1x1 conv)
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: int = 6):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class ChannelSpatialAttention(nn.Module):
    """
    Lightweight attention module combining channel and spatial attention.
    
    Channel attention: Focuses on 'what' is important
    Spatial attention: Focuses on 'where' is important
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # For attention map extraction
        self.last_channel_attention = None
        self.last_spatial_attention = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # Spatial attention  
        sa = self.spatial_attention(x_ca)
        x_out = x_ca * sa
        
        # Store for visualization
        self.last_channel_attention = ca
        self.last_spatial_attention = sa
        
        return x_out
    
    def get_attention_maps(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the last computed attention maps"""
        return self.last_channel_attention, self.last_spatial_attention


class CrossMagFusionLight(nn.Module):
    """
    Lightweight cross-magnification fusion module.
    
    Learns optimal weighting of features from different magnifications
    and combines them through attention mechanism.
    """
    def __init__(self, feat_dim: int, num_mags: int = 4):
        super().__init__()
        
        # Attention weights for each magnification
        self.attention = nn.Sequential(
            nn.Linear(feat_dim * num_mags, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_mags),
            nn.Softmax(dim=1)
        )
        
        # Feature fusion network
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * num_mags, feat_dim * 2),
            nn.BatchNorm1d(feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Increased from 0.2 to 0.3
            nn.Linear(feat_dim * 2, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(0.2)  # Added additional dropout after final layer
        )
        
        # Learnable residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
        # For visualization
        self.last_attention_weights = None
        
    def forward(self, mag_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Stack features in consistent order
        mags = ['40', '100', '200', '400']
        feats = [mag_features[mag] for mag in mags]
        stacked = torch.stack(feats, dim=1)  # [B, 4, feat_dim]
        concat = torch.cat(feats, dim=1)      # [B, 4*feat_dim]
        
        # Compute attention weights
        weights = self.attention(concat)      # [B, 4]
        self.last_attention_weights = weights
        
        # Weighted combination
        weighted = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # [B, feat_dim]
        
        # Feature fusion
        fused = self.fusion(concat)
        
        # Residual connection with learnable weight
        output = self.residual_weight * weighted + (1 - self.residual_weight) * fused
        
        return output
    
    def get_attention_weights(self) -> torch.Tensor:
        """Return the last computed attention weights"""
        return self.last_attention_weights


class MultiMagLightweightCNN(nn.Module):
    """
    Lightweight Multi-Magnification CNN for Histopathology Classification.
    
    Processes four magnifications (40x, 100x, 200x, 400x) with shared shallow
    features and magnitude-specific deep features, combined through attention.
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        base_channels: Base number of channels (default: 24)
        dropout: Dropout rate for classifier (default: 0.3)
        num_blocks_per_mag: Dictionary mapping magnifications to number of blocks
    """
    
    def __init__(
        self, 
        num_classes: int = 2,
        base_channels: int = 24,
        dropout: float = 0.4,  # Increased from 0.3 to 0.4 for better regularization
        num_blocks_per_mag: Optional[Dict[str, int]] = None
    ):
        super().__init__()
        
        self.magnifications = ['40', '100', '200', '400']
        
        # Default number of blocks per magnification
        if num_blocks_per_mag is None:
            num_blocks_per_mag = {
                '40': 2,   # Lower magnification = fewer blocks
                '100': 2,
                '200': 3,  # Higher magnification = more blocks
                '400': 3
            }
        
        # Shared shallow feature extractor
        self.shared_stem = nn.Sequential(
            # Initial convolution
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU6(inplace=True),
            
            # Depthwise separable block
            nn.Conv2d(base_channels, base_channels, 3, stride=1, padding=1, 
                     groups=base_channels, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU6(inplace=True),
            
            # Expand channels
            nn.Conv2d(base_channels, base_channels * 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU6(inplace=True)
        )
        
        # Magnitude-specific processing branches
        self.mag_branches = nn.ModuleDict()
        for mag in self.magnifications:
            self.mag_branches[mag] = self._make_mag_branch(
                base_channels * 2, 
                base_channels * 4,
                num_blocks=num_blocks_per_mag[mag]
            )
        
        # Attention modules for each magnification
        self.mag_attention = nn.ModuleDict({
            mag: ChannelSpatialAttention(base_channels * 4)
            for mag in self.magnifications
        })
        
        # Cross-magnification fusion
        self.fusion = CrossMagFusionLight(base_channels * 4, num_mags=4)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 4),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Increased from dropout (0.3) to 0.5
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),  # Increased from dropout * 0.5 (0.15) to 0.4
            nn.Linear(base_channels * 2, num_classes)
        )
        
        # For multi-task learning (optional)
        self.aux_classifier = None
        
    def _make_mag_branch(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        """Create magnitude-specific processing branch"""
        layers = []
        
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1  # Only first block downsamples
            input_channels = in_channels if i == 0 else out_channels
            
            layers.append(
                InvertedResidual(
                    input_channels,
                    out_channels,
                    stride=stride,
                    expand_ratio=6
                )
            )
            
        return nn.Sequential(*layers)
    
    def forward(
        self, 
        images_dict: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            images_dict: Dictionary with keys 'mag_40', 'mag_100', 'mag_200', 'mag_400'
            mask: Optional mask tensor (unused in lightweight model, for compatibility)
            return_features: If True, also return intermediate features
            
        Returns:
            logits: Classification logits [B, num_classes]
            features: (optional) Dictionary of intermediate features
        """
        
        # Extract shared shallow features
        shared_features = {}
        for mag in self.magnifications:
            shared_features[mag] = self.shared_stem(images_dict[f'mag_{mag}'])
        
        # Process through magnitude-specific branches
        mag_features = {}
        spatial_sizes = {}
        
        for mag in self.magnifications:
            # Magnitude-specific processing
            feat = self.mag_branches[mag](shared_features[mag])
            
            # Store spatial size before pooling (for visualization)
            spatial_sizes[mag] = feat.shape[-2:]
            
            # Apply attention
            feat = self.mag_attention[mag](feat)
            
            # Global average pooling
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            mag_features[mag] = feat
        
        # Cross-magnification fusion
        fused = self.fusion(mag_features)
        
        # Classification
        logits = self.classifier(fused)
        
        if return_features:
            features = {
                'shared': shared_features,
                'mag_specific': mag_features,
                'fused': fused,
                'spatial_sizes': spatial_sizes
            }
            return logits, features
            
        return logits
    
    @torch.no_grad()
    def get_attention_maps(self, images_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps for visualization.
        
        Returns dictionary containing:
        - channel_attention: Channel attention for each magnification
        - spatial_attention: Spatial attention for each magnification  
        - fusion_weights: Cross-magnification fusion weights
        """
        self.eval()
        
        # Forward pass to populate attention maps
        _ = self.forward(images_dict)
        
        attention_data = {
            'channel_attention': {},
            'spatial_attention': {},
            'fusion_weights': None
        }
        
        # Extract attention maps from each magnification
        for mag in self.magnifications:
            ca, sa = self.mag_attention[mag].get_attention_maps()
            attention_data['channel_attention'][mag] = ca
            attention_data['spatial_attention'][mag] = sa
        
        # Extract fusion weights
        attention_data['fusion_weights'] = self.fusion.get_attention_weights()
        
        return attention_data
    
    def get_magnification_importance(self, dataloader=None, device="cuda"):
        """Get magnification importance scores for compatibility with main.py"""
        self.eval()
        all_weights = []
        
        with torch.no_grad():
            if dataloader is not None:
                for batch_idx, (images_dict, mask, _) in enumerate(dataloader):
                    if batch_idx >= 10:  # Sample from first 10 batches
                        break
                    
                    # Move to device
                    images_dict = {k: v.to(device) for k, v in images_dict.items()}
                    
                    # Forward pass to get fusion weights
                    _ = self.forward(images_dict)
                    fusion_weights = self.fusion.get_attention_weights()
                    
                    if fusion_weights is not None:
                        all_weights.append(fusion_weights.cpu())
            
            if all_weights:
                mean_weights = torch.cat(all_weights, dim=0).mean(dim=0)
                importance_dict = {}
                for i, mag in enumerate(['40', '100', '200', '400']):
                    importance_dict[mag] = float(mean_weights[i])
                return importance_dict
            else:
                # Return default uniform weights if no data
                return {'40': 0.25, '100': 0.25, '200': 0.25, '400': 0.25}
    
    def get_model_info(self) -> Dict[str, int]:
        """Get model information including parameter counts"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count parameters by component
        stem_params = sum(p.numel() for p in self.shared_stem.parameters())
        branch_params = sum(p.numel() for p in self.mag_branches.parameters())
        attention_params = sum(p.numel() for p in self.mag_attention.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'stem_parameters': stem_params,
            'branch_parameters': branch_params,
            'attention_parameters': attention_params,
            'fusion_parameters': fusion_params,
            'classifier_parameters': classifier_params
        }
    
    def freeze_backbone(self):
        """Freeze the shared stem and magnitude branches for fine-tuning"""
        for param in self.shared_stem.parameters():
            param.requires_grad = False
        for param in self.mag_branches.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True


# Compatibility with existing codebase
def create_lightweight_model(num_classes=2, **kwargs):
    """Factory function for creating the lightweight model"""
    return MultiMagLightweightCNN(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Test the model
    model = MultiMagLightweightCNN(num_classes=2, base_channels=24)
    
    # Print model info
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value:,}")
    
    # Test forward pass
    batch_size = 4
    test_input = {
        'mag_40': torch.randn(batch_size, 3, 224, 224),
        'mag_100': torch.randn(batch_size, 3, 224, 224),
        'mag_200': torch.randn(batch_size, 3, 224, 224),
        'mag_400': torch.randn(batch_size, 3, 224, 224)
    }
    
    # Normal forward
    output = model(test_input)
    print(f"\nOutput shape: {output.shape}")
    
    # Forward with features
    output, features = model(test_input, return_features=True)
    print(f"\nWith features - Output shape: {output.shape}")
    print(f"Fused features shape: {features['fused'].shape}")
    
    # Get attention maps
    attention_maps = model.get_attention_maps(test_input)
    print(f"\nFusion weights shape: {attention_maps['fusion_weights'].shape}")