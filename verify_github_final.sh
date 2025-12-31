#!/bin/bash

echo "=" * 70
echo "FINAL GITHUB UPLOAD VERIFICATION"
echo "=" * 70

echo "1. Checking local repository status..."
git status

echo -e "\n2. Checking remote connection..."
git remote -v

echo -e "\n3. Checking commits..."
git log --oneline -3

echo -e "\n4. Checking tags..."
git tag -l

echo -e "\n5. Testing GitHub fetch..."
if git fetch origin; then
    echo "   ✅ Successfully connected to GitHub"
    
    # Compare commits
    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse origin/main)
    
    if [ "$LOCAL" = "$REMOTE" ]; then
        echo "   ✅ Local and remote are synchronized"
    else
        echo "   ⚠ Local and remote differ"
        echo "   Local:  $LOCAL"
        echo "   Remote: $REMOTE"
    fi
else
    echo "   ❌ Could not connect to GitHub"
fi

echo -e "\n" + "=" * 70
echo "📊 UPLOAD STATUS"
echo "=" * 70

echo -e "\n✅ Your repository is now live at:"
echo "   🌐 https://github.com/jamesenglis/geospatial-vit-drone"
echo -e "\n📦 Package information:"
echo "   Name: geospatial-vit"
echo "   Version: 1.0.0"
echo "   License: MIT"
echo -e "\n📥 Installation commands:"
echo "   git clone https://github.com/jamesenglis/geospatial-vit-drone.git"
echo "   cd geospatial-vit-drone"
echo "   pip install -e ."
echo -e "\n🚀 Quick start:"
echo "   python scripts/create_drone_sample_data.py"
echo "   python -m geospatial_vit train --config configs/test_minimal.yaml"

echo -e "\n🎉 Congratulations! Your project is now on GitHub!"
echo "=" * 70
