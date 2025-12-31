"""
Minimal training script
"""
import torch
import yaml

def train(config_path):
    """Simple training function"""
    print(f"Starting training with config: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Config loaded:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Create model
    from . import GeospatialViT
    model = GeospatialViT(
        img_size=config.get('img_size', 128),
        patch_size=config.get('patch_size', 16),
        in_channels=config.get('in_channels', 3),
        num_classes=config.get('num_classes', 5),
        embed_dim=config.get('model', {}).get('embed_dim', 192),
        depth=config.get('model', {}).get('depth', 2),
        num_heads=config.get('model', {}).get('num_heads', 4)
    )
    
    print(f"Model created: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Mock training loop
    print("\nMock training for 2 epochs...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    for epoch in range(config.get('training', {}).get('epochs', 2)):
        # Mock loss
        loss = 1.0 / (epoch + 1)
        print(f"Epoch {epoch + 1}: loss = {loss:.4f}")
    
    print("\nTraining complete!")
    print("Model saved to: experiments/test/model.pth")
    
    # Save dummy checkpoint
    import os
    os.makedirs('experiments/test', exist_ok=True)
    torch.save({
        'epoch': config.get('training', {}).get('epochs', 2),
        'model_state_dict': model.state_dict(),
        'config': config
    }, 'experiments/test/model.pth')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    train(args.config)
