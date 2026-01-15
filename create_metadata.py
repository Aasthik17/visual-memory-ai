# create_metadata.py
import pandas as pd
from pathlib import Path
import os

def parse_lamem_splits():
    """
    Parse LaMem split files and create unified metadata CSV.
    """
    splits_dir = Path("data/lamem/splits")
    images_dir = Path("data/lamem/images")
    
    all_data = []
    
    # Process all split files
    split_files = {
        'train': list(splits_dir.glob('train_*.txt')),
        'val': list(splits_dir.glob('val_*.txt')),
        'test': list(splits_dir.glob('test_*.txt'))
    }
    
    print("Processing LaMem splits...")
    
    for split_type, files in split_files.items():
        print(f"\nProcessing {split_type} files: {len(files)} files found")
        
        for file_path in files:
            print(f"  Reading {file_path.name}...")
            
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    # LaMem format is typically: image_path memorability_score
                    # Example: "images/1.jpg 0.567"
                    parts = line.split()
                    
                    if len(parts) >= 2:
                        # Extract image name (remove 'images/' prefix if present)
                        img_path = parts[0]
                        if img_path.startswith('images/'):
                            img_name = img_path.replace('images/', '')
                        else:
                            img_name = img_path
                        
                        # Get memorability score
                        try:
                            mem_score = float(parts[1])
                        except ValueError:
                            print(f"    Warning: Invalid score in {file_path.name} line {line_num}: {parts[1]}")
                            continue
                        
                        # Verify image exists
                        if (images_dir / img_name).exists():
                            all_data.append({
                                'image_name': img_name,
                                'memorability': mem_score,
                                'split': split_type
                            })
                        else:
                            print(f"    Warning: Image not found: {img_name}")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Remove duplicates (keep first occurrence)
    df_unique = df.drop_duplicates(subset=['image_name'], keep='first')
    
    if len(df) != len(df_unique):
        print(f"\nRemoved {len(df) - len(df_unique)} duplicate entries")
    
    # Save to CSV
    output_path = "data/lamem/metadata.csv"
    df_unique.to_csv(output_path, index=False)
    
    # Print statistics
    print(f"\n{'='*50}")
    print(f"Metadata creation complete!")
    print(f"{'='*50}")
    print(f"Total images: {len(df_unique)}")
    print(f"\nSplit distribution:")
    print(df_unique['split'].value_counts().to_string())
    print(f"\nMemorability statistics:")
    print(f"  Mean: {df_unique['memorability'].mean():.4f}")
    print(f"  Std:  {df_unique['memorability'].std():.4f}")
    print(f"  Min:  {df_unique['memorability'].min():.4f}")
    print(f"  Max:  {df_unique['memorability'].max():.4f}")
    print(f"\nSaved to: {output_path}")
    
    return df_unique

if __name__ == "__main__":
    df = parse_lamem_splits()