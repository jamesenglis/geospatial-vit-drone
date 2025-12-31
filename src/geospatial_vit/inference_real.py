"""
Inference script for trained Geospatial ViT
"""
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import yaml

class GeospatialViTInference:
    def __init__(self, checkpoint_path):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.config = checkpoint['config']
        
        # Setup device
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Create model
        from . import GeospatialViT
        self.model = GeospatialViT(
            img_size=self.config['data']['img_size'],
            patch_size=self.config['data']['patch_size'],
            in_channels=self.config['data']['in_channels'],
            num_classes=self.config['data']['num_classes'],
            embed_dim=self.config['model']['embed_dim'],
            depth=self.config['model']['depth'],
            num_heads=self.config['model']['num_heads'],
            mlp_ratio=self.config['model']['mlp_ratio'],
            dropout=self.config['model']['dropout'],
            use_geo=self.config['model']['use_geo_encoding']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Class names (you can customize these)
        self.class_names = [
            'residential', 'commercial', 'agriculture', 'forest', 'water',
            'road', 'parking', 'construction', 'grassland', 'industrial'
        ]
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Trained for {checkpoint['epoch']} epochs")
        print(f"Validation accuracy: {checkpoint.get('accuracy', 'unknown'):.2f}%")
    
    def preprocess_image(self, image_path, geo_coords=None):
        """Preprocess image for inference"""
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize
        img_size = self.config['data']['img_size']
        img = img.resize((img_size, img_size))
        
        # Convert to numpy and normalize
        img_np = np.array(img) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = (img_np - mean) / std
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Prepare geo coords if provided
        if geo_coords is not None and self.config['model']['use_geo_encoding']:
            geo_tensor = torch.tensor(geo_coords, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            geo_tensor = None
        
        return img_tensor, geo_tensor
    
    def predict(self, image_path, geo_coords=None):
        """Make prediction on single image"""
        # Preprocess
        image_tensor, geo_tensor = self.preprocess_image(image_path, geo_coords)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor, geo_tensor, task='classification')
            probabilities = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, pred_class].item()
        
        return {
            'class_id': pred_class,
            'class_name': self.class_names[pred_class] if pred_class < len(self.class_names) else f'Class_{pred_class}',
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy()
        }
    
    def predict_batch(self, image_paths, geo_coords_list=None):
        """Make predictions on batch of images"""
        results = []
        for i, img_path in enumerate(image_paths):
            geo_coords = geo_coords_list[i] if geo_coords_list else None
            result = self.predict(img_path, geo_coords)
            result['image_path'] = str(img_path)
            results.append(result)
        return results
    
    def visualize_prediction(self, image_path, prediction):
        """Simple visualization of prediction"""
        from PIL import Image, ImageDraw, ImageFont
        import matplotlib.pyplot as plt
        
        # Load image
        img = Image.open(image_path)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Prediction info
        axes[1].text(0.1, 0.5, 
                    f"Prediction: {prediction['class_name']}\n"
                    f"Confidence: {prediction['confidence']*100:.1f}%\n"
                    f"Class ID: {prediction['class_id']}",
                    fontsize=12,
                    verticalalignment='center')
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--geo-coords', type=float, nargs=3, 
                       help='Geospatial coordinates (lat lon alt)')
    parser.add_argument('--visualize', action='store_true')
    
    args = parser.parse_args()
    
    # Run inference
    inference = GeospatialViTInference(args.checkpoint)
    result = inference.predict(args.image, args.geo_coords)
    
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Image: {args.image}")
    print(f"Predicted class: {result['class_name']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"Class ID: {result['class_id']}")
    print("\nAll probabilities:")
    for i, prob in enumerate(result['probabilities']):
        class_name = inference.class_names[i] if i < len(inference.class_names) else f'Class_{i}'
        print(f"  {class_name}: {prob*100:.2f}%")
    
    if args.visualize:
        fig = inference.visualize_prediction(args.image, result)
        output_path = Path(args.image).stem + '_prediction.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\nVisualization saved to: {output_path}")

if __name__ == "__main__":
    main()
