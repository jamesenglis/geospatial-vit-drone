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
        if self.config.get('hardware', {}).get('device') == 'auto':
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config['hardware']['device'])
        
        print(f"Using device: {self.device}")
        
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
        
        # Setup mixed precision
        self.mixed_precision = self.config['training']['mixed_precision']
        self.scaler = GradScaler(enabled=self.mixed_precision and self.device.type == 'mps')
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs']
        )
        
        # Create output directory
        self.output_dir = Path(self.config['logging']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
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
        
        return total_loss / len(train_loader), 100. * correct / total
    
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
        
        print(f"\nStarting training for {self.config['training']['epochs']} epochs")
        print(f"Output directory: {self.output_dir}")
        
        for epoch in range(self.config['training']['epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint('best_model.pth', epoch, val_acc)
                print(f"  ✅ Saved best model (acc: {val_acc:.2f}%)")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_acc)
        
        # Training complete
        training_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Training complete!")
        print(f"Best validation accuracy: {best_acc:.2f}%")
        print(f"Total training time: {training_time:.2f} seconds")
        print(f"Models saved in: {self.output_dir}")
        print(f"{'='*50}")
    
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
    
    config = yaml.safe_load(open(config_path, 'r'))
    
    train_loader, val_loader = create_drone_dataloaders(
        data_dir=config['data']['data_dir'],
        annotations_file=Path(config['data']['data_dir']) / "annotations.csv",
        batch_size=config['data']['batch_size'],
        img_size=config['data']['img_size']
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
