#!/bin/bash

echo "🔧 Testing GitHub Installation Instructions"
echo "=========================================="

# Create a temporary directory for testing
TEST_DIR="test_installation_$(date +%s)"
mkdir "$TEST_DIR"
cd "$TEST_DIR"

echo "1. Cloning repository from GitHub..."
git clone https://github.com/jamesenglis/geospatial-vit-drone.git
cd geospatial-vit-drone

echo "2. Creating conda environment..."
conda create -n test-geospatial-vit python=3.9 -y

echo "3. Activating conda environment..."
conda activate test-geospatial-vit

echo "4. Installing package..."
pip install -e .

echo "5. Testing installation..."
python -c "
import sys
print('Testing imports...')
try:
    import geospatial_vit
    print('✅ geospatial_vit imported successfully')
    
    from geospatial_vit import GeospatialViT
    print('✅ GeospatialViT imported successfully')
    
    # Test model creation
    model = GeospatialViT(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10
    )
    print(f'✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters')
    
    # Test PyTorch
    import torch
    print(f'✅ PyTorch version: {torch.__version__}')
    print(f'✅ MPS available: {torch.backends.mps.is_available()}')
    
    print('\n🎉 All tests passed! Installation successful.')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"

echo "6. Cleaning up..."
cd ../..
rm -rf "$TEST_DIR"
conda deactivate
conda env remove -n test-geospatial-vit -y

echo "\n✅ Installation test complete!"
