"""
Generate test data for Geospatial ViT
"""
import numpy as np
from PIL import Image
import os
from pathlib import Path
import pandas as pd

def generate_sample_data(num_images=10, size=224):
    """Generate sample drone-like images"""
    output_dir = Path("data/raw/sample_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create classes
    classes = ['urban', 'agriculture', 'forest', 'water', 'infrastructure']
    
    for i in range(num_images):
        # Generate different patterns for different classes
        class_idx = i % len(classes)
        
        if classes[class_idx] == 'urban':
            # Urban: grid pattern
            img = np.zeros((size, size, 3), dtype=np.uint8)
            for x in range(0, size, 20):
                img[x:x+10, :, :] = 150  # Roads
            for y in range(0, size, 25):
                img[:, y:y+15, :] = 200  # Buildings
                
        elif classes[class_idx] == 'agriculture':
            # Agriculture: field patterns
            img = np.random.randint(100, 150, (size, size, 3), dtype=np.uint8)
            # Add some field boundaries
            for x in range(0, size, 40):
                img[x:x+2, :, :] = 50
            for y in range(0, size, 40):
                img[:, y:y+2, :] = 50
                
        elif classes[class_idx] == 'forest':
            # Forest: green texture
            img = np.random.randint(50, 100, (size, size, 3), dtype=np.uint8)
            img[:, :, 1] = np.random.randint(100, 200, (size, size))  # More green
            
        elif classes[class_idx] == 'water':
            # Water: blue texture
            img = np.random.randint(50, 100, (size, size, 3), dtype=np.uint8)
            img[:, :, 2] = np.random.randint(150, 255, (size, size))  # More blue
            
        elif classes[class_idx] == 'infrastructure':
            # Infrastructure: lines and squares
            img = np.random.randint(100, 200, (size, size, 3), dtype=np.uint8)
            # Add some infrastructure patterns
            img[size//2-20:size//2+20, size//2-20:size//2+20, :] = 255
            
        # Save as PNG
        img_pil = Image.fromarray(img)
        img_path = output_dir / f"drone_image_{i:03d}_{classes[class_idx]}.png"
        img_pil.save(img_path)
        
        print(f"Generated: {img_path}")
    
    # Create a simple annotation file
    annotations = []
    for i in range(num_images):
        class_idx = i % len(classes)
        annotations.append({
            'image': f"drone_image_{i:03d}_{classes[class_idx]}.png",
            'class': class_idx,
            'class_name': classes[class_idx],
            'lat': 40.0 + i * 0.01,  # Fake coordinates
            'lon': -74.0 + i * 0.01,
            'alt': 100.0
        })
    
    # Save as CSV
    df = pd.DataFrame(annotations)
    df.to_csv(output_dir / "annotations.csv", index=False)
    
    print(f"\nGenerated {num_images} sample images in {output_dir}")
    print("Classes:", classes)

if __name__ == "__main__":
    generate_sample_data(num_images=20)
