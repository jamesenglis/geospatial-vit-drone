#!/bin/bash

echo "=" * 70
echo "FINAL COMPREHENSIVE TEST"
echo "=" * 70

echo "1. Testing imports..."
python -c "
import geospatial_vit
from geospatial_vit import GeospatialViT
import torch
print('✅ All imports work')
"

echo -e "\n2. Testing CLI..."
python -m geospatial_vit --version

echo -e "\n3. Checking data..."
if [ -d "data/drone_samples" ]; then
    echo "✅ Data exists"
    echo "   Samples: $(ls data/drone_samples/*.png 2>/dev/null | wc -l) images"
else
    echo "⚠ No data, generating..."
    python scripts/create_drone_sample_data.py
fi

echo -e "\n4. Running training test..."
python -m geospatial_vit train --config configs/simple_test.yaml

echo -e "\n" + "=" * 70
echo "🎉 TEST COMPLETE!"
echo "=" * 70
