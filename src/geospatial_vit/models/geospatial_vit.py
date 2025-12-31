import torch
import torch.nn as nn
import torch.nn.functional as F

class GeospatialViT(nn.Module):
    """Vision Transformer for geospatial drone imagery"""
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        use_geo=True
    ):
        super().__init__()
        
        # Store parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Simple convolutional backbone for testing
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classifier
        self.classifier = nn.Linear(256, num_classes)
        
        # Geospatial encoder (optional)
        self.use_geo = use_geo
        if use_geo:
            self.geo_encoder = nn.Sequential(
                nn.Linear(3, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 256)
            )
        else:
            self.geo_encoder = None
            
    def forward(self, x, geo_coords=None, task='classification'):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            geo_coords: Geospatial coordinates (B, 3) [lat, lon, alt], optional
            task: Task type ('classification', 'detection', 'segmentation')
        
        Returns:
            Output tensor
        """
        # Extract features
        features = self.features(x)
        features = features.flatten(1)  # (B, 256)
        
        # Add geospatial features if provided
        if self.use_geo and geo_coords is not None and self.geo_encoder is not None:
            geo_features = self.geo_encoder(geo_coords)
            features = features + geo_features
        
        # Classification
        output = self.classifier(features)
        
        return output
    
    def get_attention_maps(self, x, geo_coords=None):
        """Get attention maps (placeholder for now)"""
        # Return dummy attention maps for compatibility
        B, C, H, W = x.shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        return [torch.randn(B, self.num_heads, num_patches + 1, num_patches + 1)]
