print("=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

import torch
print(f"1. PyTorch: {torch.__version__}")
print(f"   MPS available: {torch.backends.mps.is_available()}")

import sys
sys.path.insert(0, 'src')

try:
    import geospatial_vit
    print("2. ✅ Package imported")
    
    from geospatial_vit import GeospatialViT
    print("3. ✅ GeospatialViT imported")
    
    model = GeospatialViT(
        img_size=128,
        patch_size=16,
        in_channels=3,
        num_classes=5
    )
    print(f"4. ✅ Model created ({sum(p.numel() for p in model.parameters()):,} params)")
    
    x = torch.randn(1, 3, 128, 128)
    output = model(x)
    print(f"5. ✅ Forward pass: {output.shape}")
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! Project is ready.")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Generate sample data: python scripts/generate_test_data.py")
    print("2. Run training: python -m geospatial_vit train --config configs/test_minimal.yaml")
    print("3. Check experiments/test/ for results")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
