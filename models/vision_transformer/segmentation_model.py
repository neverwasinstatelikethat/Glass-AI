"""
Vision Transformer for Defect Segmentation
Implements U-Net style segmentation with Vision Transformer backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatchMerging(nn.Module):
    """Merge patches to reduce spatial dimensions and increase channels"""
    
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        out_channels = out_channels or 2 * in_channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, height, width]
            
        Returns:
            [batch_size, out_channels, height//2, width//2]
        """
        x = self.conv(x)
        # Layer norm on channels
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x


class PatchExpanding(nn.Module):
    """Expand patches to increase spatial dimensions"""
    
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        out_channels = out_channels or in_channels // 2
        self.expand = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, height, width]
            
        Returns:
            [batch_size, out_channels, height*2, width*2]
        """
        x = self.expand(x)
        x = self.pixel_shuffle(x)
        # Layer norm on channels
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x


class ViTSegmentationEncoder(nn.Module):
    """Vision Transformer Encoder for segmentation"""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24]
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        
        # Initial patch embedding
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
        
        # Positional embeddings
        n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Hierarchical feature extraction
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i, (depth, n_head) in enumerate(zip(depths, num_heads)):
            stage = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=n_head,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    batch_first=True
                ) for _ in range(depth)
            ])
            self.stages.append(stage)
            
            # Downsample between stages (except last)
            if i < len(depths) - 1:
                downsample = PatchMerging(embed_dim, embed_dim * 2)
                self.downsamples.append(downsample)
                embed_dim *= 2
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [batch_size, in_channels, img_size, img_size]
            
        Returns:
            List of feature maps at different scales
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [batch_size, n_patches, embed_dim]
        x = x + self.pos_embed
        
        # Add class token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        
        features = []
        
        # Hierarchical processing
        for i, stage in enumerate(self.stages):
            # Transformer blocks
            for block in stage:
                x = block(x)
            
            # Store features (excluding class token)
            feat = x[:, 1:, :]  # [batch_size, n_patches, embed_dim]
            features.append(feat)
            
            # Downsample (except last stage)
            if i < len(self.stages) - 1:
                # Reshape to image format for downsampling
                h = w = int(feat.shape[1] ** 0.5)
                feat_img = rearrange(feat, 'b (h w) c -> b c h w', h=h, w=w)
                feat_img = self.downsamples[i](feat_img)
                # Reshape back to sequence
                feat_flat = rearrange(feat_img, 'b c h w -> b (h w) c')
                x = torch.cat([cls_tokens, feat_flat], dim=1)  # Re-add class token
        
        return features


class ViTSegmentationDecoder(nn.Module):
    """Decoder for segmentation with skip connections"""
    
    def __init__(
        self,
        embed_dims: List[int],
        num_classes: int = 2,
        final_upscale: bool = True
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.final_upscale = final_upscale
        
        # Upsampling layers
        self.upsamples = nn.ModuleList()
        self.fuse_layers = nn.ModuleList()
        
        # Process from deepest to shallowest
        for i in reversed(range(len(embed_dims) - 1)):
            # Upsample current level
            upsample = PatchExpanding(embed_dims[i + 1], embed_dims[i])
            self.upsamples.append(upsample)
            
            # Fuse with skip connection
            fuse = nn.Sequential(
                nn.Linear(embed_dims[i] * 2, embed_dims[i]),
                nn.GELU(),
                nn.Linear(embed_dims[i], embed_dims[i])
            )
            self.fuse_layers.append(fuse)
        
        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Linear(embed_dims[0], embed_dims[0] // 2),
            nn.GELU(),
            nn.Linear(embed_dims[0] // 2, num_classes)
        )
        
        # Final upsampling to original resolution
        if final_upscale:
            self.final_upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature maps from encoder (shallow to deep)
            
        Returns:
            Segmentation map [batch_size, num_classes, H, W]
        """
        # Start from deepest feature
        x = features[-1]  # [batch_size, n_patches, embed_dim]
        
        # Process from deepest to shallowest
        for i, (upsample, fuse) in enumerate(zip(self.upsamples, self.fuse_layers)):
            # Get corresponding skip connection
            skip_idx = len(features) - 2 - i
            skip = features[skip_idx]  # [batch_size, n_patches_skip, embed_dim_skip]
            
            # Upsample current feature
            # Reshape to image format
            h_curr = w_curr = int(x.shape[1] ** 0.5)
            x_img = rearrange(x, 'b (h w) c -> b c h w', h=h_curr, w=w_curr)
            x_up = upsample(x_img)  # [batch_size, embed_dim_skip, h_up, w_up]
            
            # Reshape skip connection to match
            h_skip = w_skip = int(skip.shape[1] ** 0.5)
            if h_skip != x_up.shape[2] or w_skip != x_up.shape[3]:
                # Resize skip connection if needed
                skip_img = rearrange(skip, 'b (h w) c -> b c h w', h=h_skip, w=w_skip)
                skip_img = F.interpolate(skip_img, size=(x_up.shape[2], x_up.shape[3]), 
                                       mode='bilinear', align_corners=False)
                skip = rearrange(skip_img, 'b c h w -> b (h w) c')
            
            # Fuse with skip connection
            x_up_flat = rearrange(x_up, 'b c h w -> b (h w) c')
            x = torch.cat([x_up_flat, skip], dim=2)  # Concat along channel dimension
            x = fuse(x)  # [batch_size, n_patches, embed_dim_skip]
        
        # Final segmentation
        seg_logits = self.seg_head(x)  # [batch_size, n_patches, num_classes]
        
        # Reshape to image format
        h_out = w_out = int(seg_logits.shape[1] ** 0.5)
        seg_map = rearrange(seg_logits, 'b (h w) c -> b c h w', h=h_out, w=w_out)
        
        # Final upscale if needed
        if self.final_upscale:
            seg_map = F.interpolate(seg_map, scale_factor=2, mode='bilinear', align_corners=False)
        
        return seg_map


class ViTUNetSegmentation(nn.Module):
    """
    U-Net style segmentation with Vision Transformer backbone
    """
    
    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 2,
        embed_dim: int = 768,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24]
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = ViTSegmentationEncoder(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads
        )
        
        # Decoder
        embed_dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.decoder = ViTSegmentationDecoder(
            embed_dims=embed_dims,
            num_classes=num_classes
        )
        
        logger.info(f"Initialized ViTUNetSegmentation with {num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [batch_size, in_channels, img_size, img_size]
            
        Returns:
            Segmentation map [batch_size, num_classes, img_size, img_size]
        """
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        seg_map = self.decoder(features)
        
        return seg_map


def create_vit_segmentation_model(
    img_size: int = 224,
    in_channels: int = 3,
    num_classes: int = 2,
    **kwargs
) -> ViTUNetSegmentation:
    """
    Factory function for creating ViT segmentation model
    
    Args:
        img_size: Input image size
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        **kwargs: Other model parameters
        
    Returns:
        ViTUNetSegmentation model
    """
    model = ViTUNetSegmentation(
        img_size=img_size,
        in_channels=in_channels,
        num_classes=num_classes,
        **kwargs
    )
    
    return model


if __name__ == "__main__":
    # Test model
    model = create_vit_segmentation_model(
        img_size=224,
        in_channels=3,
        num_classes=5
    )
    
    # Test data
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    seg_output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Segmentation output shape: {seg_output.shape}")
    
    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")