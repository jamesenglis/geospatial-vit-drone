import sys
sys.path.append('src')

from geospatial_vit.dataloaders.drone_loader import create_drone_dataloaders

print("Debugging data loading...")
train_loader, val_loader = create_drone_dataloaders(
    data_dir="data/drone_samples",
    batch_size=2,
    img_size=128
)

# Get a batch
batch = next(iter(train_loader))
print(f"\nBatch shape: {batch['image'].shape}")
print(f"Labels: {batch['label']}")
print(f"Label range: {batch['label'].min()} to {batch['label'].max()}")
print(f"Unique labels: {torch.unique(batch['label'])}")

# Check annotations
import pandas as pd
import os
if os.path.exists("data/drone_samples/annotations.csv"):
    df = pd.read_csv("data/drone_samples/annotations.csv")
    print(f"\nAnnotations file has {len(df)} entries")
    print(f"Class distribution:")
    print(df['class_id'].value_counts().sort_index())
else:
    print("\nNo annotations.csv found")
