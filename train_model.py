# train_model.py
import torch
import pandas as pd
from pathlib import Path

from data.dataloader import create_dataloaders
from models.model import create_model, count_parameters
from models.train import Trainer

def main():
    # Configuration
    DATA_DIR = "./data/lamem"
    IMAGES_DIR = "./data/lamem/images"
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4  # Adjust based on your CPU cores
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"{'='*60}")
    print(f"Visual Memory AI - Training Pipeline")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}\n")
    
    # Load metadata
    print("Loading dataset...")
    df = pd.read_csv(f"{DATA_DIR}/metadata.csv")
    print(f"Total images: {len(df)}")
    
    # Use pre-defined splits from LaMem
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_df):,} images ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} images ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} images ({len(test_df)/len(df)*100:.1f}%)")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        IMAGES_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model('resnet50', pretrained=True, device=DEVICE)
    total_params = count_parameters(model)
    print(f"  Architecture: ResNet50")
    print(f"  Parameters: {total_params:,}")
    print(f"  Model size: {total_params * 4 / 1e6:.2f} MB")
    
    # Create directories
    Path('./checkpoints').mkdir(exist_ok=True)
    Path('./logs').mkdir(exist_ok=True)
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        checkpoint_dir='./checkpoints',
        log_dir='./logs'
    )
    
    history = trainer.train(
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=10
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Best validation Pearson: {max(history['val_pearson']):.4f}")
    print(f"\nCheckpoints saved to: ./checkpoints/")
    print(f"TensorBoard logs saved to: ./logs/")
    print(f"\nTo view training curves, run:")
    print(f"  tensorboard --logdir=./logs")

if __name__ == "__main__":
    main()