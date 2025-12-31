import numpy as np
from PIL import Image
import os
from pathlib import Path
import pandas as pd
import json

def create_realistic_drone_images(output_dir="data/drone_samples", num_images=50):
    """Create realistic drone-like images with different land cover types"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Land cover classes for drone imagery
    classes = [
        'residential',      # Residential buildings
        'commercial',       # Commercial buildings
        'agriculture',      # Farmland
        'forest',           # Forest/trees
        'water',            # Water bodies
        'road',             # Roads/highways
        'parking',          # Parking lots
        'construction',     # Construction sites
        'grassland',        # Grass/parks
        'industrial'        # Industrial areas
    ]
    
    annotations = []
    
    for i in range(num_images):
        # Assign class
        class_idx = i % len(classes)
        class_name = classes[class_idx]
        
        # Create image based on class
        img_size = 512  # Typical drone image size
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        if class_name == 'residential':
            # Residential: grid of houses
            img[:, :, :] = 180  # Light gray background
            # Add houses
            for x in range(50, img_size, 100):
                for y in range(50, img_size, 100):
                    # House body
                    img[x:x+30, y:y+40, 0] = 200  # Red roof
                    img[x:x+30, y:y+40, 1] = 150  # Green tint
                    img[x:x+30, y:y+40, 2] = 100  # Blue tint
                    
        elif class_name == 'agriculture':
            # Agriculture: field patterns
            img[:, :, 1] = np.random.randint(100, 200, (img_size, img_size))  # Green fields
            img[:, :, 0] = np.random.randint(50, 150, (img_size, img_size))   # Brown earth
            # Add field boundaries
            for x in range(0, img_size, 64):
                img[x:x+2, :, :] = 50  # Dark boundaries
            for y in range(0, img_size, 64):
                img[:, y:y+2, :] = 50
                
        elif class_name == 'forest':
            # Forest: green with tree patterns
            base_green = np.random.randint(50, 100, (img_size, img_size))
            img[:, :, 1] = base_green + np.random.randint(0, 50, (img_size, img_size))
            # Add tree crowns (circles)
            for _ in range(20):
                cx = np.random.randint(20, img_size-20)
                cy = np.random.randint(20, img_size-20)
                radius = np.random.randint(10, 30)
                # Draw circle
                y, x = np.ogrid[-radius:radius, -radius:radius]
                mask = x*x + y*y <= radius*radius
                img[cx-radius:cx+radius, cy-radius:cy+radius, 1][mask] = 200
                
        elif class_name == 'water':
            # Water: blue with wave patterns
            img[:, :, 2] = np.random.randint(100, 200, (img_size, img_size))  # Blue water
            # Add wave patterns
            for x in range(0, img_size, 20):
                img[x:x+2, :, 2] = np.random.randint(150, 255, img_size)
                
        elif class_name == 'road':
            # Road: gray with lane markings
            img[:, :, :] = 100  # Gray asphalt
            # Add road lines
            for y in range(0, img_size, 40):
                img[:, y:y+5, :] = 255  # White lines
            # Add center line
            img[img_size//2-2:img_size//2+2, :, 0] = 255  # Yellow center
                
        else:
            # Generic pattern for other classes
            img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
            # Add class-specific tint
            if class_name == 'commercial':
                img[:, :, 0] = np.clip(img[:, :, 0] + 50, 0, 255)  # More red
            elif class_name == 'industrial':
                img[:, :, :] = np.clip(img[:, :, :] - 50, 0, 255)  # Darker
        
        # Add some noise for realism
        noise = np.random.randint(-20, 20, (img_size, img_size, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save image
        img_pil = Image.fromarray(img)
        filename = f"drone_{i:04d}_{class_name}.png"
        img_path = output_path / filename
        img_pil.save(img_path)
        
        # Create annotation
        # Generate fake GPS coordinates (somewhere in New York area)
        lat = 40.7128 + (np.random.random() - 0.5) * 0.1
        lon = -74.0060 + (np.random.random() - 0.5) * 0.1
        alt = 100 + np.random.random() * 50  # 100-150m altitude
        
        annotations.append({
            'image_id': i,
            'filename': filename,
            'class_id': class_idx,
            'class_name': class_name,
            'latitude': float(lat),
            'longitude': float(lon),
            'altitude': float(alt),
            'image_size': f"{img_size}x{img_size}",
            'date_captured': '2024-01-01',
            'sensor_type': 'RGB',
            'ground_sample_distance': 0.05  # 5cm/pixel
        })
        
        if i % 10 == 0:
            print(f"Created {i+1}/{num_images} images...")
    
    # Save annotations
    df = pd.DataFrame(annotations)
    df.to_csv(output_path / "annotations.csv", index=False)
    
    # Also save as JSON for easy reading
    with open(output_path / "annotations.json", 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\n✅ Created {num_images} drone images in {output_path}")
    print(f"   Classes: {classes}")
    print(f"   Annotations saved to: {output_path}/annotations.csv")
    
    return output_path

if __name__ == "__main__":
    create_realistic_drone_images(num_images=100)
