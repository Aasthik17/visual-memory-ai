"""
Data preprocessing module for Visual Memory AI.
Handles LaMem dataset download, extraction, and preparation.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple, List
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


class LaMemPreprocessor:
    """Preprocessor for the LaMem (Large-scale Image Memorability) dataset."""
    
    def __init__(self, data_dir: str = "./data/lamem"):
        """
        Initialize preprocessor.
        
        Args:
            data_dir: Root directory for dataset storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.images_dir = self.data_dir / "images"
        self.splits_dir = self.data_dir / "splits"
        
    def download_dataset(self) -> None:
        """
        Download LaMem dataset.
        Note: This is a placeholder. Real implementation would download from:
        http://memorability.csail.mit.edu/download.html
        """
        print("Dataset download instructions:")
        print("1. Visit: http://memorability.csail.mit.edu/download.html")
        print("2. Download lamem.tar.gz")
        print(f"3. Extract to: {self.images_dir}")
        print("4. Download splits from the same page")
        
    def create_metadata(self, splits_file: str) -> pd.DataFrame:
        """
        Create metadata DataFrame from splits file.
        
        Args:
            splits_file: Path to splits/annotations file
            
        Returns:
            DataFrame with image paths and memorability scores
        """
        # Example format for LaMem:
        # image_name, memorability_score, split
        if not os.path.exists(splits_file):
            raise FileNotFoundError(f"Splits file not found: {splits_file}")
            
        df = pd.read_csv(splits_file)
        return df
    
    def validate_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that all images exist and are readable.
        
        Args:
            df: DataFrame with image information
            
        Returns:
            Filtered DataFrame with valid images only
        """
        valid_indices = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating images"):
            img_path = self.images_dir / row['image_name']
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_indices.append(idx)
            except Exception as e:
                print(f"Invalid image {img_path}: {e}")
                
        return df.loc[valid_indices].reset_index(drop=True)
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/val/test sets.
        
        Args:
            df: Full dataset DataFrame
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        np.random.seed(random_seed)
        shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_df = shuffled[:train_end]
        val_df = shuffled[train_end:val_end]
        test_df = shuffled[val_end:]
        
        return train_df, val_df, test_df
    
    def compute_statistics(self, df: pd.DataFrame) -> dict:
        """
        Compute dataset statistics.
        
        Args:
            df: Dataset DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_images': len(df),
            'mean_memorability': df['memorability'].mean(),
            'std_memorability': df['memorability'].std(),
            'min_memorability': df['memorability'].min(),
            'max_memorability': df['memorability'].max(),
            'median_memorability': df['memorability'].median()
        }
        return stats


def create_sample_dataset(output_dir: str = "./data/lamem", n_samples: int = 1000) -> None:
    """
    Create a sample dataset for testing (synthetic data).
    Use this for development when LaMem dataset is not available.
    
    Args:
        output_dir: Directory to save sample data
        n_samples: Number of synthetic samples to create
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic images and metadata
    metadata = []
    
    for i in tqdm(range(n_samples), desc="Creating sample dataset"):
        # Create random image
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        img_name = f"sample_{i:05d}.jpg"
        img.save(images_dir / img_name)
        
        # Generate memorability score (normally distributed around 0.5)
        memorability = np.clip(np.random.normal(0.5, 0.15), 0, 1)
        
        metadata.append({
            'image_name': img_name,
            'memorability': memorability
        })
    
    # Save metadata
    df = pd.DataFrame(metadata)
    df.to_csv(output_path / "metadata.csv", index=False)
    print(f"Sample dataset created at {output_path}")
    print(f"Total images: {len(df)}")
    print(f"Mean memorability: {df['memorability'].mean():.3f}")


if __name__ == "__main__":
    # Example usage
    preprocessor = LaMemPreprocessor()
    
    # For development: create sample dataset
    create_sample_dataset(n_samples=100)
    
    print("Preprocessing setup complete!")