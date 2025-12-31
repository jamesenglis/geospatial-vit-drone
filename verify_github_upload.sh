#!/bin/bash

echo "🔍 Verifying GitHub Upload"
echo "=========================="

# Check remote URL
echo "1. Checking remote URL:"
git remote -v

# Check commits
echo -e "\n2. Checking commits:"
git log --oneline -5

# Check tags
echo -e "\n3. Checking tags:"
git tag -l

# Check if we can fetch from remote
echo -e "\n4. Testing connection to GitHub:"
git fetch origin

if [ $? -eq 0 ]; then
    echo "   ✅ Successfully connected to GitHub"
    
    # Compare local and remote
    LOCAL_HASH=$(git rev-parse main)
    REMOTE_HASH=$(git rev-parse origin/main)
    
    if [ "$LOCAL_HASH" = "$REMOTE_HASH" ]; then
        echo "   ✅ Local and remote are in sync"
    else
        echo "   ⚠ Local and remote differ"
        echo "   Local:  $LOCAL_HASH"
        echo "   Remote: $REMOTE_HASH"
    fi
else
    echo "   ❌ Could not connect to GitHub"
fi

# Check what would be pushed
echo -e "\n5. Files to be pushed:"
git status

echo -e "\n📊 Summary:"
echo "Repository URL: https://github.com/jamesenglis/geospatial-vit-drone"
echo "To view online: https://github.com/jamesenglis/geospatial-vit-drone"
