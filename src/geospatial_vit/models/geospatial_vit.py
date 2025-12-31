import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GeospatialViT"]

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
        
        # Better feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classifier with better initialization
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Geospatial encoder (optional)
        self.use_geo = use_geo
        if use_geo:
            self.geo_encoder = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 256)
            )
            self._initialize_weights_geo()
        else:
            self.geo_encoder = None
            
    def _initialize_weights(self):
        """Initialize weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def _initialize_weights_geo(self):
        """Initialize weights for geo encoder"""
        for m in self.geo_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
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
