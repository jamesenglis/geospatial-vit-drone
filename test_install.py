import sys
print("Testing installation...")

# Try to import
try:
    # Add src to path just in case
    sys.path.insert(0, 'src')
    from geospatial_vit import GeospatialViT
    print("✅ Successfully imported GeospatialViT")
    
    # Create model
    model = GeospatialViT(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10
    )
    print(f"✅ Model created: {model.__class__.__name__}")
    
    # Test forward pass
    import torch
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"✅ Forward pass successful. Output shape: {output.shape}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
    # Check what's in src
    import os
    print("\nChecking src directory contents:")
    if os.path.exists("src"):
        for root, dirs, files in os.walk("src"):
            level = root.replace("src", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files:
                if file.endswith(".py"):
                    print(f"{subindent}{file}")
