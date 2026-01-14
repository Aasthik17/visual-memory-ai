"""
DataLoader module for Visual Memory AI.
Implements PyTorch Dataset and efficient data loading.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Callable


class MemorabilityDataset(Dataset):
    """PyTorch Dataset for image memorability prediction."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        transform: Optional[Callable] = None,
        normalize_scores: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            df: DataFrame with 'image_name' and 'memorability' columns
            images_dir: Directory containing images
            transform: Optional transform to apply to images
            normalize_scores: Whether to normalize memorability scores to [0, 1]
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.normalize_scores = normalize_scores
        
        if normalize_scores:
            self.score_min = df['memorability'].min()
            self.score_max = df['memorability'].max()
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Index of item
            
        Returns:
            Tuple of (image_tensor, memorability_score)
        """
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.images_dir / row['image_name']
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get memorability score
        score = row['memorability']
        if self.normalize_scores:
            score = (score - self.score_min) / (self.score_max - self.score_min)
        
        score = torch.tensor(score, dtype=torch.float32)
        
        return image, score
    
    def denormalize_score(self, normalized_score: float) -> float:
        """
        Convert normalized score back to original scale.
        
        Args:
            normalized_score: Score in [0, 1]
            
        Returns:
            Score in original scale
        """
        if not self.normalize_scores:
            return normalized_score
        return normalized_score * (self.score_max - self.score_min) + self.score_min


def get_transforms(is_training: bool = True) -> transforms.Compose:
    """
    Get image transformations for training or inference.
    
    Args:
        is_training: Whether transforms are for training
        
    Returns:
        Composed transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    images_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        images_dir: Directory containing images
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = MemorabilityDataset(
        train_df,
        images_dir,
        transform=get_transforms(is_training=True),
        normalize_scores=True
    )
    
    val_dataset = MemorabilityDataset(
        val_df,
        images_dir,
        transform=get_transforms(is_training=False),
        normalize_scores=True
    )
    
    test_dataset = MemorabilityDataset(
        test_df,
        images_dir,
        transform=get_transforms(is_training=False),
        normalize_scores=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Load sample data
    df = pd.read_csv("./data/lamem/metadata.csv")
    
    # Split data
    from data.preprocess import LaMemPreprocessor
    preprocessor = LaMemPreprocessor()
    train_df, val_df, test_df = preprocessor.split_data(df)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df,
        val_df,
        test_df,
        "./data/lamem/images",
        batch_size=16
    )
    
    # Test loading
    images, scores = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")