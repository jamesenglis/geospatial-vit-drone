"""
Real training pipeline for Geospatial ViT
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import yaml
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np

class GeospatialViTTrainer:
    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("✅ Using MPS (Apple Silicon GPU)")
        else:
            self.device = torch.device('cpu')
            print("⚠ Using CPU (MPS not available)")
        
        # Get config values with flexible structure
        if 'data' in self.config:
            data_config = self.config['data']
            img_size = data_config.get('img_size', 224)
            patch_size = data_config.get('patch_size', 16)
            in_channels = data_config.get('in_channels', 3)
            num_classes = data_config.get('num_classes', 10)
            data_dir = data_config.get('data_dir', 'data/drone_samples')
            batch_size = data_config.get('batch_size', 8)
        else:
            img_size = self.config.get('img_size', 224)
            patch_size = self.config.get('patch_size', 16)
            in_channels = self.config.get('in_channels', 3)
            num_classes = self.config.get('num_classes', 10)
            data_dir = self.config.get('data_dir', 'data/drone_samples')
            batch_size = self.config.get('batch_size', 8)
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        
        # Model config with defaults
        if 'model' in self.config:
            model_config = self.config['model']
        else:
            model_config = self.config
        
        embed_dim = model_config.get('embed_dim', 768)
        depth = model_config.get('depth', 12)
        num_heads = model_config.get('num_heads', 12)
        mlp_ratio = model_config.get('mlp_ratio', 4.0)
        dropout = model_config.get('dropout', 0.1)
        use_geo = model_config.get('use_geo_encoding', model_config.get('use_geo', True))
        
        # Training config with defaults
        if 'training' in self.config:
            train_config = self.config['training']
        else:
            train_config = self.config
        
        self.epochs = train_config.get('epochs', 50)
        self.lr = train_config.get('lr', 0.001)
        self.weight_decay = train_config.get('weight_decay', 0.05)
        self.mixed_precision = train_config.get('mixed_precision', True)
        self.save_interval = train_config.get('save_interval', 5)
        
        # Create model
        from . import GeospatialViT
        self.model = GeospatialViT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_geo=use_geo
        ).to(self.device)
        
        # Setup mixed precision
        self.scaler = GradScaler(enabled=self.mixed_precision and self.device.type == 'mps')
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs
        )
        
        # Create output directory
        if 'logging' in self.config:
            logging_config = self.config['logging']
            output_dir = logging_config.get('output_dir', 'experiments/default')
            experiment_name = logging_config.get('experiment_name', 'geospatial_vit')
        else:
            output_dir = self.config.get('output_dir', 'experiments/default')
            experiment_name = self.config.get('experiment_name', 'geospatial_vit')
        
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        
        print(f"✅ Model initialized")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Image size: {img_size}, Batch size: {batch_size}")
        print(f"   Num classes: {num_classes}")
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            geo_coords = batch.get('geo_coords', None)
            if geo_coords is not None:
                geo_coords = geo_coords.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward with mixed precision if enabled
            with autocast(enabled=self.mixed_precision and self.device.type == 'mps'):
                outputs = self.model(images, geo_coords, task='classification')
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            if self.mixed_precision and self.device.type == 'mps':
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Calculate metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{current_lr:.6f}'
            })
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                geo_coords = batch.get('geo_coords', None)
                if geo_coords is not None:
                    geo_coords = geo_coords.to(self.device)
                
                outputs = self.model(images, geo_coords, task='classification')
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader):
        best_acc = 0
        start_time = time.time()
        
        print(f"\n🚀 Starting training for {self.epochs} epochs")
        print(f"   Device: {self.device}")
        print(f"   Mixed precision: {self.mixed_precision}")
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Validation samples: {len(val_loader.dataset)}")
        print(f"   Learning rate: {self.lr}")
        
        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1}/{self.epochs}:")
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint('best_model.pth', epoch, val_acc)
                print(f"   💾 Saved best model (acc: {val_acc:.2f}%)")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_acc)
        
        # Training complete
        training_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"🎉 Training complete!")
        print(f"   Best validation accuracy: {best_acc:.2f}%")
        print(f"   Total training time: {training_time:.2f} seconds")
        print(f"   Models saved in: {self.output_dir}")
        print(f"{'='*60}")
        
        # Save final model
        self.save_checkpoint('final_model.pth', self.epochs-1, best_acc)
    
    def save_checkpoint(self, filename, epoch, acc):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': acc,
            'config': self.config
        }
        
        torch.save(checkpoint, self.output_dir / filename)

def main(config_path):
    # Create dataloaders
    from .dataloaders.drone_loader import create_drone_dataloaders
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get data config with flexible structure
    if 'data' in config:
        data_config = config['data']
        data_dir = data_config.get('data_dir', 'data/drone_samples')
        batch_size = data_config.get('batch_size', 8)
        img_size = data_config.get('img_size', 256)
    else:
        data_dir = config.get('data_dir', 'data/drone_samples')
        batch_size = config.get('batch_size', 8)
        img_size = config.get('img_size', 256)
    
    print(f"📂 Loading data from: {data_dir}")
    print(f"   Batch size: {batch_size}, Image size: {img_size}")
    
    train_loader, val_loader = create_drone_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size
    )
    
    # Create trainer and train
    trainer = GeospatialViTTrainer(config_path)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)
