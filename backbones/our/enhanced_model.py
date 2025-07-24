import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class StableCrossAttention(nn.Module):
    """Stable cross-attention mechanism with careful initialization"""
    
    def __init__(self, feat_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        
        # Ensure divisibility
        assert feat_dim % num_heads == 0, f"feat_dim {feat_dim} must be divisible by num_heads {num_heads}"
        
        # Linear layers for Q, K, V
        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(feat_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        
        # Stable initialization
        self._init_weights()
    
    def _init_weights(self):
        # Xavier uniform initialization with smaller scale
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.5)
        
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.constant_(self.out_proj.bias, 0)
    
    def forward(self, x):
        B, M, D = x.shape  # Batch, Magnifications, Dimension
        
        # Apply layer norm first for stability
        x_norm = self.norm(x)
        
        # Compute Q, K, V
        Q = self.q_proj(x_norm).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_norm).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with temperature scaling
        scale = (self.head_dim ** -0.5) * 0.5  # Additional scaling for stability
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Clamp attention scores for stability
        attn_scores = torch.clamp(attn_scores, -10, 10)
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, M, D)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # Residual connection with scaling
        output = x + 0.1 * output  # Small residual scaling for stability
        
        return output


class StableMagnificationFusion(nn.Module):
    """Stable magnification fusion with learned importance weights"""
    
    def __init__(self, feat_dim, num_mags=4, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_mags = num_mags
        
        # Magnification importance network
        self.importance_net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.LayerNorm(feat_dim // 4, eps=1e-5),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 4, 1)
        )
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim, eps=1e-5),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Final fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(feat_dim * num_mags, feat_dim * 2),
            nn.LayerNorm(feat_dim * 2, eps=1e-5),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, features_list):
        # Stack features: [B, num_mags, feat_dim]
        stacked_features = torch.stack(features_list, dim=1)
        B, M, D = stacked_features.shape
        
        # Transform features
        transformed = self.feature_transform(stacked_features)
        
        # Compute importance weights
        importance_scores = self.importance_net(transformed)  # [B, M, 1]
        importance_weights = F.softmax(importance_scores, dim=1)
        
        # Weighted features
        weighted_features = transformed * importance_weights
        
        # Global average pooling (attention-based)
        global_feat = weighted_features.sum(dim=1)  # [B, D]
        
        # Concatenation-based fusion
        concat_feat = stacked_features.view(B, -1)  # [B, M*D]
        fused_feat = self.fusion_layer(concat_feat)
        
        # Combine global and fused features
        final_feat = 0.6 * global_feat + 0.4 * fused_feat
        
        return final_feat, importance_weights.squeeze(-1)


class EnhancedMMNet(nn.Module):
    """Enhanced MMNet with stable attention and fusion mechanisms"""
    
    def __init__(self, magnifications=['40', '100', '200', '400'], num_classes=2, dropout=0.3, backbone='resnet18'):
        super().__init__()
        self.magnifications = magnifications
        self.num_mags = len(magnifications)
        
        # Shared backbone for stability
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool='avg')
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            self.feat_dim = self.backbone(dummy).shape[1]
        
        # Cross-magnification attention
        self.cross_attention = StableCrossAttention(
            feat_dim=self.feat_dim,
            num_heads=max(1, self.feat_dim // 128),  # Adaptive number of heads
            dropout=dropout * 0.5
        )
        
        # Magnification fusion
        self.mag_fusion = StableMagnificationFusion(
            feat_dim=self.feat_dim,
            num_mags=self.num_mags,
            dropout=dropout * 0.5
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, self.feat_dim // 2),
            nn.LayerNorm(self.feat_dim // 2, eps=1e-5),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(self.feat_dim // 2, num_classes)
        )
        
        # Initialize classifier
        self._init_classifier()
        
        print(f"EnhancedMMNet: feat_dim={self.feat_dim}, num_heads={max(1, self.feat_dim // 128)}")
    
    def _init_classifier(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, images_dict, mask=None):
        # Extract features for each magnification
        features = []
        for mag in self.magnifications:
            x = images_dict[f'mag_{mag}']
            feat = self.backbone(x)  # [B, feat_dim]
            
            # Stability check
            feat = torch.clamp(feat, -10, 10)
            features.append(feat)
        
        # Stack for cross-attention: [B, num_mags, feat_dim]
        stacked_features = torch.stack(features, dim=1)
        
        # Apply cross-magnification attention
        attended_features = self.cross_attention(stacked_features)
        
        # Convert back to list for fusion
        attended_list = [attended_features[:, i, :] for i in range(self.num_mags)]
        
        # Magnification fusion
        fused_features, importance_weights = self.mag_fusion(attended_list)
        
        # Final stability check
        fused_features = torch.clamp(fused_features, -5, 5)
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Final logits clamping
        logits = torch.clamp(logits, -10, 10)
        
        return logits
    
    def get_magnification_importance(self, dataloader=None, device="cuda"):
        """Get magnification importance from the fusion layer"""
        if dataloader is None:
            return {mag: 1.0/len(self.magnifications) for mag in self.magnifications}
        
        self.eval()
        all_weights = []
        
        with torch.no_grad():
            for images_dict, mask, _ in dataloader:
                images = {k: v.to(device) for k, v in images_dict.items()}
                
                # Extract features
                features = []
                for mag in self.magnifications:
                    feat = self.backbone(images[f'mag_{mag}'])
                    features.append(feat)
                
                # Get importance weights
                _, importance_weights = self.mag_fusion(features)
                all_weights.append(importance_weights.cpu())
        
        # Average weights across all samples
        mean_weights = torch.cat(all_weights, dim=0).mean(dim=0)
        return {mag: float(mean_weights[i]) for i, mag in enumerate(self.magnifications)}