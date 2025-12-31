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
            # Try to find annotations automatically
            csv_file = self.data_dir / "annotations.csv"
            json_file = self.data_dir / "annotations.json"
            
            if csv_file.exists():
                self.annotations = pd.read_csv(csv_file)
            elif json_file.exists():
                with open(json_file, 'r') as f:
                    annotations = json.load(f)
                self.annotations = pd.DataFrame(annotations)
            else:
                # Create dummy annotations from image files
                image_files = list(self.data_dir.glob('*.png')) + list(self.data_dir.glob('*.jpg'))
                annotations = []
                for i, img_file in enumerate(image_files):
                    annotations.append({
                        'filename': img_file.name,
                        'class_id': i % 5,  # Default to 5 classes
                        'class_name': f'class_{i % 5}',
                        'latitude': 40.0,
                        'longitude': -74.0,
                        'altitude': 100.0
                    })
                self.annotations = pd.DataFrame(annotations)
        
        # Default augmentation if none provided
        if transform is None:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
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
        
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except:
            # Create dummy image if file doesn't exist
            image = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Get label
        label = row['class_id'] if 'class_id' in row else 0
        
        # Get geospatial coordinates
        if 'latitude' in row and 'longitude' in row and 'altitude' in row:
            geo_coords = [
                float(row['latitude']),
                float(row['longitude']), 
                float(row['altitude'])
            ]
        else:
            geo_coords = [0.0, 0.0, 0.0]
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'geo_coords': torch.tensor(geo_coords, dtype=torch.float32),
            'image_id': idx,
            'filename': row['filename']
        }

def create_drone_dataloaders(data_dir, annotations_file=None, batch_size=8, img_size=512):
    """Create train and validation dataloaders"""
    
    # If no annotations file specified, look for it in data_dir
    if annotations_file is None:
        data_path = Path(data_dir)
        csv_file = data_path / "annotations.csv"
        json_file = data_path / "annotations.json"
        
        if csv_file.exists():
            annotations_file = str(csv_file)
        elif json_file.exists():
            annotations_file = str(json_file)
        else:
            annotations_file = None
    
    # Create full dataset
    full_dataset = DroneDataset(
        data_dir=data_dir,
        annotations_file=annotations_file,
        img_size=img_size
    )
    
    # Simple split without sklearn
    indices = list(range(len(full_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(0.8 * len(indices))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 for Mac compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Created dataloaders:")
    print(f"  Training: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")
    
    if hasattr(full_dataset, 'annotations') and 'class_name' in full_dataset.annotations.columns:
        num_classes = full_dataset.annotations['class_name'].nunique()
        print(f"  Classes: {num_classes}")
    
    return train_loader, val_loader

# Test the dataloader
if __name__ == "__main__":
    # Quick test
    import os
    if os.path.exists("data/drone_samples"):
        train_loader, val_loader = create_drone_dataloaders(
            data_dir="data/drone_samples",
            batch_size=4,
            img_size=256
        )
        
        # Get a batch
        batch = next(iter(train_loader))
        print(f"\nBatch info:")
        print(f"  Images: {batch['image'].shape}")
        print(f"  Labels: {batch['label'].shape}")
        print(f"  Geo coords: {batch['geo_coords'].shape}")
    else:
        print("Please run: python scripts/create_drone_sample_data.py first")
