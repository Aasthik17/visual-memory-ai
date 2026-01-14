"""
Training module for Visual Memory AI.
Implements training loop with metrics tracking and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr


class Trainer:
    """Trainer class for memorability prediction model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device for training
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float('inf')
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        predictions = []
        targets = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, scores in pbar:
            images = images.to(self.device)
            scores = scores.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, scores)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions.extend(outputs.detach().cpu().numpy())
            targets.extend(scores.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            self.writer.add_scalar('train/batch_loss', loss.item(), self.global_step)
            self.global_step += 1
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        correlation, _ = pearsonr(predictions, targets)
        
        metrics = {
            'loss': avg_loss,
            'pearson': correlation,
            'mse': np.mean((np.array(predictions) - np.array(targets)) ** 2)
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for images, scores in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                scores = scores.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, scores)
                
                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                targets.extend(scores.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        correlation, _ = pearsonr(predictions, targets)
        
        metrics = {
            'loss': avg_loss,
            'pearson': correlation,
            'mse': np.mean((np.array(predictions) - np.array(targets)) ** 2)
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'global_step': self.global_step
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {metrics['loss']:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Epochs to wait before early stopping
            
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'train_pearson': [],
            'val_loss': [],
            'val_pearson': []
        }
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Pearson: {train_metrics['pearson']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                  f"Pearson: {val_metrics['pearson']:.4f}")
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Log to tensorboard
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/train_pearson', train_metrics['pearson'], epoch)
            self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/val_pearson', val_metrics['pearson'], epoch)
            self.writer.add_scalar('epoch/lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save history
            history['train_loss'].append(train_metrics['loss'])
            history['train_pearson'].append(train_metrics['pearson'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_pearson'].append(val_metrics['pearson'])
            
            # Checkpointing
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        self.writer.close()
        return history


if __name__ == "__main__":
    from models.model import create_model
    from data.dataloader import create_dataloaders
    import pandas as pd
    from data.preprocess import LaMemPreprocessor
    
    # Load data
    df = pd.read_csv("./data/lamem/metadata.csv")
    preprocessor = LaMemPreprocessor()
    train_df, val_df, test_df = preprocessor.split_data(df)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        "./data/lamem/images",
        batch_size=16
    )
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model('resnet50', device=device)
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, device=device)
    history = trainer.train(num_epochs=50)
    
    print("Training complete!")