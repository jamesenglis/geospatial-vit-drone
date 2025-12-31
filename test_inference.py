import torch
import sys
sys.path.append('src')

from geospatial_vit import GeospatialViT
from geospatial_vit.dataloaders.drone_loader import DroneDataset
from torch.utils.data import DataLoader

print("Testing inference on trained model...")

# Load a trained model if exists
checkpoint_path = "experiments/improved_test/geospatial_vit/best_model.pth"
if not torch.backends.mps.is_available():
    checkpoint_path = "experiments/improved_test/geospatial_vit/best_model.pth"

try:
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Accuracy: {checkpoint['accuracy']:.2f}%")
    
    # Create model
    config = checkpoint['config']
    model = GeospatialViT(
        img_size=config.get('img_size', 128),
        patch_size=16,
        in_channels=3,
        num_classes=config.get('num_classes', 10)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test on sample data
    dataset = DroneDataset(
        data_dir="data/drone_samples",
        annotations_file="data/drone_samples/annotations.csv",
        img_size=128
    )
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    
    with torch.no_grad():
        outputs = model(batch['image'])
        predictions = torch.softmax(outputs, dim=1)
        
        print(f"\n📊 Sample predictions:")
        for i in range(min(2, len(batch['label']))):
            true_label = batch['label'][i].item()
            pred_label = torch.argmax(predictions[i]).item()
            confidence = predictions[i][pred_label].item()
            
            print(f"   Sample {i+1}: True={true_label}, Pred={pred_label}, Conf={confidence:.2%}")
            
    print("\n✅ Inference test successful!")
    
except FileNotFoundError:
    print("⚠ No trained model found. Please train first.")
    print("Run: python -m geospatial_vit train --config configs/improved_test.yaml")
except Exception as e:
    print(f"❌ Error: {e}")
