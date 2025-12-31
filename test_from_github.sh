#!/bin/bash

echo "Testing installation from GitHub instructions..."
echo "=============================================="

# Create a test directory
TEST_DIR="/tmp/test_github_install_$(date +%s)"
echo "Test directory: $TEST_DIR"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

echo "1. Cloning repository..."
git clone https://github.com/jamesenglis/geospatial-vit-drone.git
cd geospatial-vit-drone

echo "2. Checking repository structure..."
ls -la

echo "3. Creating virtual environment..."
python -m venv venv_test
source venv_test/bin/activate

echo "4. Installing package..."
pip install -e .

echo "5. Testing installation..."
python -c "
print('Testing imports...')
try:
    import geospatial_vit
    print('✅ geospatial_vit imported')
    
    from geospatial_vit import GeospatialViT
    print('✅ GeospatialViT imported')
    
    # Create model
    model = GeospatialViT(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10
    )
    print(f'✅ Model created: {sum(p.numel() for p in model.parameters()):,} params')
    
    # Test PyTorch
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    
    print('\n🎉 All tests passed! Installation works correctly.')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"

echo "6. Cleaning up..."
deactivate
cd /
rm -rf "$TEST_DIR"

echo -e "\n✅ Installation test complete!"
