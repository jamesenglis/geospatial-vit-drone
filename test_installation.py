#!/usr/bin/env python3
"""
Quick test to verify installation
"""
import torch
import sys
import os

def test_mps():
    """Test MPS support on Apple Silicon"""
    print("Testing Apple Silicon MPS support...")
    if torch.backends.mps.is_available():
        print("✅ MPS is available!")
        
        # Test a simple operation
        device = torch.device("mps")
        x = torch.ones(2, 3, device=device)
        y = torch.ones(2, 3, device=device)
        z = x + y
        
        if torch.all(z == 2):
            print("✅ MPS tensor operations work correctly")
        else:
            print("❌ MPS tensor operations failed")
            
        return True
    else:
        print("❌ MPS is not available")
        print("This might be because:")
        print("1. You're not on Apple Silicon (M1/M2)")
        print("2. PyTorch wasn't built with MPS support")
        print("3. You need to update PyTorch")
        return False

def test_imports():
    """Test that we can import all necessary modules"""
    print("\nTesting imports...")
    
    modules = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('rasterio', 'Rasterio'),
    ]
    
    all_imported = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name}")
            all_imported = False
            
    return all_imported

def test_project_structure():
    """Verify project structure exists"""
    print("\nChecking project structure...")
    
    required_dirs = [
        'src/geospatial_vit',
        'configs',
        'data/raw',
        'experiments',
        'notebooks',
        'tests'
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ (missing)")
            all_exist = False
            
    return all_exist

def main():
    print("=" * 60)
    print("Geospatial ViT Installation Test")
    print("=" * 60)
    
    # Test MPS
    mps_ok = test_mps()
    
    # Test imports
    imports_ok = test_imports()
    
    # Test structure
    structure_ok = test_project_structure()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if mps_ok and imports_ok and structure_ok:
        print("✅ All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Generate sample data: python scripts/generate_test_data.py")
        print("2. Run tests: python -m pytest tests/")
        print("3. Train a test model: python -m geospatial_vit train --config configs/m1_test.yaml")
    else:
        print("❌ Some tests failed.")
        print("\nTroubleshooting:")
        if not mps_ok:
            print("- For MPS issues, try: pip install --upgrade torch torchvision")
        if not imports_ok:
            print("- Install missing packages: pip install -r requirements.txt")
        if not structure_ok:
            print("- Some directories are missing. Run the setup script again.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
