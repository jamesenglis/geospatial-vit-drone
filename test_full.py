#!/usr/bin/env python
"""
Complete test for Geospatial ViT
"""
import os
import sys
import torch

print("=" * 60)
print("Geospatial ViT - Complete Installation Test")
print("=" * 60)

# Test 1: Environment
print("\n1. Environment:")
print(f"   Python: {sys.version}")
print(f"   PyTorch: {torch.__version__}")
print(f"   MPS available: {torch.backends.mps.is_available()}")

# Test 2: Package structure
print("\n2. Package structure:")
project_root = os.getcwd()
print(f"   Project root: {project_root}")

# Check if setup.py exists
setup_py = os.path.join(project_root, "setup.py")
if os.path.exists(setup_py):
    print("   ✅ setup.py exists")
else:
    print("   ❌ setup.py missing")

# Check src directory
src_dir = os.path.join(project_root, "src")
if os.path.exists(src_dir):
    print("   ✅ src/ directory exists")
    
    # List Python files
    py_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                rel_path = os.path.relpath(os.path.join(root, file), src_dir)
                py_files.append(rel_path)
    
    print(f"   Found {len(py_files)} Python files in src/")
    for file in sorted(py_files)[:5]:  # Show first 5
        print(f"     - {file}")
    if len(py_files) > 5:
        print(f"     ... and {len(py_files) - 5} more")
else:
    print("   ❌ src/ directory missing")

# Test 3: Try imports
print("\n3. Testing imports:")

# Method 1: Direct import
print("   Method 1 - Direct import:")
try:
    import geospatial_vit
    print("     ✅ geospatial_vit package imported")
    print(f"     Version: {getattr(geospatial_vit, '__version__', 'unknown')}")
except ImportError as e:
    print(f"     ❌ Failed: {e}")

# Method 2: With src in path
print("\n   Method 2 - With src in path:")
sys.path.insert(0, src_dir)
try:
    import geospatial_vit as gsvit
    print("     ✅ Imported with src path")
    
    # Check what's available
    available = [x for x in dir(gsvit) if not x.startswith('_')]
    print(f"     Available: {available}")
    
    # Try to get GeospatialViT
    if hasattr(gsvit, 'GeospatialViT'):
        print("     ✅ GeospatialViT class found")
    elif 'GeospatialViT' in available:
        print("     ✅ GeospatialViT in available list")
    else:
        print("     ❌ GeospatialViT not found")
        
except ImportError as e:
    print(f"     ❌ Failed: {e}")

# Test 4: Create model
print("\n4. Creating model:")
try:
    # First try to get GeospatialViT from the package
    if 'gsvit' in locals():
        GeospatialViT = getattr(gsvit, 'GeospatialViT', None)
        if GeospatialViT is None:
            # Try to import directly
            from geospatial_vit.models.geospatial_vit import GeospatialViT
    
    # Create model
    model = GeospatialViT(
        img_size=128,
        patch_size=16,
        in_channels=3,
        num_classes=5,
        embed_dim=192,
        depth=2,
        num_heads=4
    )
    print(f"   ✅ Model created")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    x = torch.randn(2, 3, 128, 128, device=device)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"   ✅ Forward pass successful")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Device: {device}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
