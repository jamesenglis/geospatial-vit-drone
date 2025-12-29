"""
Verify that the Geospatial ViT setup is working correctly
"""
import torch
import sys
import os

print("=" * 60)
print("Geospatial ViT - Setup Verification")
print("=" * 60)

# Check Python version
print(f"Python version: {sys.version}")

# Check PyTorch
print(f"\nPyTorch version: {torch.__version__}")

# Check MPS availability
print(f"MPS (Apple Silicon GPU) available: {torch.backends.mps.is_available()}")
if torch.backends.mps.is_available():
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
# Check CUDA (should be False on Mac)
print(f"CUDA available: {torch.cuda.is_available()}")

# Test basic tensor operations
print("\nTesting tensor operations on MPS...")
try:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.randn(2, 3, 224, 224, device=device)
        y = torch.randn(2, 3, 224, 224, device=device)
        z = x + y
        print(f"✓ Tensor operations work on MPS")
        print(f"  Tensor shape: {z.shape}")
        print(f"  Tensor device: {z.device}")
    else:
        print("⚠ MPS not available, using CPU")
        device = torch.device("cpu")
        x = torch.randn(2, 3, 224, 224)
        y = torch.randn(2, 3, 224, 224)
        z = x + y
        print(f"✓ Tensor operations work on CPU")
except Exception as e:
    print(f"✗ Tensor operations failed: {e}")

# Test model import
print("\nTesting model imports...")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
    from geospatial_vit.models.geospatial_vit import GeospatialViT
    print("✓ Successfully imported GeospatialViT")
    
    # Try to create a model
    model = GeospatialViT(
        img_size=224,
        patch_size=16,
        in_channels=4,
        num_classes=10,
        embed_dim=384,  # Smaller for testing
        depth=4,
        num_heads=6
    )
    print(f"✓ Successfully created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    test_input = torch.randn(1, 4, 224, 224)
    if torch.backends.mps.is_available():
        model = model.to("mps")
        test_input = test_input.to("mps")
    
    with torch.no_grad():
        output = model(test_input, task='classification')
        print(f"✓ Model forward pass works")
        print(f"  Output shape: {output.shape}")
        
except ImportError as e:
    print(f"✗ Import failed: {e}")
except Exception as e:
    print(f"✗ Model test failed: {e}")

# Check installed packages
print("\nChecking key packages...")
required_packages = ['torch', 'torchvision', 'numpy', 'rasterio', 'geopandas']
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg} installed")
    except ImportError:
        print(f"✗ {pkg} NOT installed")

print("\n" + "=" * 60)
print("Setup verification complete!")
print("=" * 60)
