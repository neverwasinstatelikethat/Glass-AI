"""
Vision Transformer Feature Extractor
Extracts deep features from images for downstream tasks like clustering, retrieval, and anomaly detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import logging
from typing import List, Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViTFeaturePyramid(nn.Module):
    """
    Multi-scale feature extraction using Vision Transformer
    Extracts features from different layers of the transformer
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        feature_levels: List[int] = [3, 6, 9, 12]
    ):
        """
        Args:
            img_size: Input image size
            patch_size: Patch size for tokenization
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            feature_levels: Which transformer layers to extract features from
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.feature_levels = sorted(feature_levels)
        
        # Patch embedding
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
        
        # Positional embeddings
        n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        logger.info(f"Initialized ViTFeaturePyramid with levels: {feature_levels}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-level features
        
        Args:
            x: Input image [batch_size, in_channels, img_size, img_size]
            
        Returns:
            Dict with features at different levels
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [batch_size, n_patches, embed_dim]
        
        # Add positional embeddings and class token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        features = {}
        
        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # Extract features at specified levels
            if (i + 1) in self.feature_levels:
                # Apply normalization
                feat = self.norm(x)
                features[f'level_{i+1}'] = feat
        
        # Global features (from class token)
        features['global'] = x[:, 0]  # [batch_size, embed_dim]
        
        # Local features (from patch tokens)
        features['local'] = x[:, 1:]  # [batch_size, n_patches, embed_dim]
        
        return features


class MultiScaleFeatureFusion(nn.Module):
    """
    Fuses features from different scales with attention weighting
    """
    
    def __init__(self, embed_dim: int = 768, num_levels: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        
        # Attention weights for different levels
        self.level_weights = nn.Parameter(torch.ones(num_levels))
        
        # Projection layers to unify dimensions
        self.projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_levels)
        ])
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * num_levels, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-scale features
        
        Args:
            features: List of features from different levels
            
        Returns:
            Fused feature vector [batch_size, embed_dim]
        """
        # Normalize attention weights
        weights = F.softmax(self.level_weights, dim=0)
        
        # Project and weight features
        weighted_features = []
        for i, (feat, proj) in enumerate(zip(features, self.projections)):
            # Global average pooling for each level
            if len(feat.shape) == 3:  # [batch, seq_len, embed_dim]
                global_feat = feat.mean(dim=1)  # [batch, embed_dim]
            else:  # [batch, embed_dim]
                global_feat = feat
            
            # Project and weight
            projected = proj(global_feat)
            weighted = projected * weights[i]
            weighted_features.append(weighted)
        
        # Concatenate and fuse
        concatenated = torch.cat(weighted_features, dim=1)
        fused = self.fusion(concatenated)
        
        return fused


class ViTFeatureExtractor(nn.Module):
    """
    Complete feature extraction pipeline
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        feature_levels: List[int] = [3, 6, 9, 12],
        pretrained: bool = False
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        # Feature pyramid network
        self.backbone = ViTFeaturePyramid(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            feature_levels=feature_levels
        )
        
        # Feature fusion
        self.fusion = MultiScaleFeatureFusion(
            embed_dim=embed_dim,
            num_levels=len(feature_levels) + 1  # +1 for global features
        )
        
        # Dimensionality reduction for compact features
        self.compact_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4)
        )
        
        # L2 normalization for feature matching
        self.l2_norm = lambda x: F.normalize(x, p=2, dim=-1)
        
        if pretrained:
            self._load_pretrained_weights()
        
        logger.info("Initialized ViTFeatureExtractor")
    
    def _load_pretrained_weights(self):
        """Load pretrained weights (placeholder)"""
        logger.info("Loading pretrained weights...")
        # In practice, this would load ImageNet pretrained weights
        pass
    
    def forward(self, x: torch.Tensor, return_multiscale: bool = False) -> Dict[str, torch.Tensor]:
        """
        Extract features from input images
        
        Args:
            x: Input images [batch_size, in_channels, img_size, img_size]
            return_multiscale: Whether to return features at all scales
            
        Returns:
            Dict with extracted features
        """
        # Extract multi-level features
        multi_features = self.backbone(x)
        
        # Prepare features for fusion
        fusion_input = []
        level_keys = [f'level_{level}' for level in self.backbone.feature_levels]
        
        # Add global features
        fusion_input.append(multi_features['global'])
        
        # Add level features
        for key in level_keys:
            if key in multi_features:
                fusion_input.append(multi_features[key])
        
        # Fuse features
        fused_features = self.fusion(fusion_input)
        
        # Compact representation
        compact_features = self.compact_projection(fused_features)
        normalized_features = self.l2_norm(compact_features)
        
        result = {
            'features': fused_features,
            'compact': compact_features,
            'normalized': normalized_features
        }
        
        if return_multiscale:
            result['multiscale'] = multi_features
        
        return result
    
    def extract_descriptors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract normalized descriptors for matching/retrieval
        
        Args:
            x: Input images
            
        Returns:
            Normalized descriptors [batch_size, embed_dim//4]
        """
        features = self.forward(x)
        return features['normalized']


class FeatureSimilarityMatcher(nn.Module):
    """
    Matches features using cosine similarity
    """
    
    def __init__(self, feature_dim: int = 192):
        super().__init__()
        self.feature_dim = feature_dim
    
    def forward(self, query_features: torch.Tensor, 
                reference_features: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between query and reference features
        
        Args:
            query_features: [batch_size, feature_dim]
            reference_features: [num_references, feature_dim]
            
        Returns:
            Similarity scores [batch_size, num_references]
        """
        # Cosine similarity
        similarity = torch.matmul(query_features, reference_features.t())
        return similarity


def create_feature_extractor(
    img_size: int = 224,
    embed_dim: int = 768,
    **kwargs
) -> ViTFeatureExtractor:
    """
    Factory function for creating feature extractor
    
    Args:
        img_size: Input image size
        embed_dim: Embedding dimension
        **kwargs: Other parameters
        
    Returns:
        ViTFeatureExtractor model
    """
    model = ViTFeatureExtractor(
        img_size=img_size,
        embed_dim=embed_dim,
        **kwargs
    )
    
    return model


if __name__ == "__main__":
    # Test model
    model = create_feature_extractor(
        img_size=224,
        embed_dim=768,
        feature_levels=[3, 6, 9, 12]
    )
    
    # Test data
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Extract features
    features = model(x, return_multiscale=True)
    print("Extracted features:")
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: dict with keys {list(value.keys())}")
    
    # Extract descriptors
    descriptors = model.extract_descriptors(x)
    print(f"Descriptors shape: {descriptors.shape}")
    
    # Test similarity matching
    matcher = FeatureSimilarityMatcher(feature_dim=192)
    query_desc = descriptors
    ref_desc = torch.randn(10, 192)  # 10 reference descriptors
    similarity = matcher(query_desc, ref_desc)
    print(f"Similarity matrix shape: {similarity.shape}")