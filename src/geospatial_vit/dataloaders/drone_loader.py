"""
Data loader for real drone imagery with geospatial metadata
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json

class DroneDataset(Dataset):
    """Dataset for drone imagery with geospatial metadata"""
    
    def __init__(self, data_dir, annotations_file, transform=None, img_size=512):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.transform = transform
        
        # Load annotations
        if annotations_file.endswith('.csv'):
            self.annotations = pd.read_csv(annotations_file)
        elif annotations_file.endswith('.json'):
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            self.annotations = pd.DataFrame(annotations)
        else:
            raise ValueError("Annotations file must be .csv or .json")
        
        # Default augmentation if none provided
        if transform is None:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        
        # Load image
        img_path = self.data_dir / row['filename']
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Get label
        label = row['class_id']
        
        # Get geospatial coordinates
        geo_coords = [
            row['latitude'],
            row['longitude'], 
            row['altitude']
        ]
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'geo_coords': torch.tensor(geo_coords, dtype=torch.float32),
            'image_id': row['image_id'],
            'filename': row['filename']
        }

def create_drone_dataloaders(data_dir, annotations_file, batch_size=8, img_size=512):
    """Create train and validation dataloaders"""
    
    from sklearn.model_selection import train_test_split
    
    # Load annotations to split
    if annotations_file.endswith('.csv'):
        annotations = pd.read_csv(annotations_file)
    else:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        annotations = pd.DataFrame(annotations)
    
    # Split indices
    train_indices, val_indices = train_test_split(
        range(len(annotations)),
        test_size=0.2,
        random_state=42,
        stratify=annotations['class_id'] if 'class_id' in annotations.columns else None
    )
    
    # Create datasets
    train_dataset = DroneDataset(
        data_dir=data_dir,
        annotations_file=annotations_file,
        img_size=img_size
    )
    
    val_dataset = DroneDataset(
        data_dir=data_dir,
        annotations_file=annotations_file,
        img_size=img_size
    )
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Created dataloaders:")
    print(f"  Training: {len(train_subset)} images")
    print(f"  Validation: {len(val_subset)} images")
    print(f"  Classes: {annotations['class_name'].nunique() if 'class_name' in annotations.columns else 'Unknown'}")
    
    return train_loader, val_loader

# Test the dataloader
if __name__ == "__main__":
    # Quick test
    train_loader, val_loader = create_drone_dataloaders(
        data_dir="data/drone_samples",
        annotations_file="data/drone_samples/annotations.csv",
        batch_size=4,
        img_size=256
    )
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch info:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Labels: {batch['label'].shape}")
    print(f"  Geo coords: {batch['geo_coords'].shape}")
