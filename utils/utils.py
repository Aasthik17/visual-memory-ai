"""
Utility functions for Visual Memory AI.
Helper functions for configuration, logging, and visualization.
"""

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir: str = "./logs", name: str = "visual_memory_ai") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        name: Logger name
        
    Returns:
        Configured logger
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(name)
    return logger


def get_device(device_name: str = "auto") -> str:
    """
    Get device for computation.
    
    Args:
        device_name: 'cuda', 'cpu', or 'auto'
        
    Returns:
        Device string
    """
    if device_name == "auto":
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_name


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: str
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Performance metrics
        filepath: Save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> int:
    """
    Load model checkpoint.
    
    Args:
        filepath: Checkpoint file path
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        
    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch']


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training metrics
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Correlation plot
    axes[1].plot(history['train_pearson'], label='Train Pearson', marker='o')
    axes[1].plot(history['val_pearson'], label='Val Pearson', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Pearson Correlation')
    axes[1].set_title('Training & Validation Correlation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def denormalize_image(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """
    Denormalize image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor [C, H, W]
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized image as numpy array [H, W, C]
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    # Clip to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose
    image = tensor.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    
    return image


def calculate_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size information
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'total_mb': size_mb,
        'params_mb': param_size / 1024 / 1024,
        'buffers_mb': buffer_size / 1024 / 1024
    }


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_directories(config: Dict[str, Any]):
    """
    Create necessary directories from config.
    
    Args:
        config: Configuration dictionary
    """
    paths = [
        config['checkpointing']['dir'],
        config['logging']['tensorboard_dir'],
        config['paths']['outputs'],
        config['paths']['visualizations'],
        config['paths']['predictions']
    ]
    
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test config loading
    config = load_config("../config.yaml")
    print(f"Config loaded: {config['model']['architecture']}")
    
    # Test device detection
    device = get_device("auto")
    print(f"Device: {device}")
    
    # Test seed setting
    seed_everything(42)
    print("Random seed set to 42")
    
    # Test logging
    logger = setup_logging()
    logger.info("Utilities test complete!")