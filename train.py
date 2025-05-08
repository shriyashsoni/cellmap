import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from dataset import get_data_loaders
from model import CellMapModel, MultiScaleCellMapModel, CombinedLoss

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch
    """
    model.train()
    running_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get data
        images = batch['image'].to(device)
        abundances = batch['abundance'].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, abundances)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item() * images.size(0)
    
    # Calculate epoch loss
    epoch_loss = running_loss / len(dataloader.dataset)
    
    return epoch_loss

def validate(model, dataloader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Get data
            images = batch['image'].to(device)
            abundances = batch['abundance'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, abundances)
            
            # Update statistics
            running_loss += loss.item() * images.size(0)
            
            # Store predictions and targets for correlation calculation
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(abundances.cpu().numpy())
    
    # Calculate epoch loss
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Calculate Spearman correlation
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate correlation for each cell type
    correlations = []
    for i in range(all_preds.shape[1]):
        corr, _ = spearmanr(all_preds[:, i], all_targets[:, i])
        if not np.isnan(corr):
            correlations.append(corr)
    
    # Calculate mean correlation
    mean_correlation = np.mean(correlations)
    
    return epoch_loss, mean_correlation

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50, save_dir='checkpoints'):
    """
    Train the model
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize variables
    best_correlation = 0.0
    train_losses = []
    val_losses = []
    val_correlations = []
    
    # Train for specified number of epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_correlation = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print statistics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Correlation: {val_correlation:.4f}")
        
        # Save model if it's the best so far
        if val_correlation > best_correlation:
            best_correlation = val_correlation
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_correlation': val_correlation
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved best model with correlation: {val_correlation:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_correlation': val_correlation
        }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Store statistics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_correlations.append(val_correlation)
        
        # Plot and save learning curves
        plot_learning_curves(train_losses, val_losses, val_correlations, save_dir)
    
    return model, train_losses, val_losses, val_correlations

def plot_learning_curves(train_losses, val_losses, val_correlations, save_dir):
    """
    Plot and save learning curves
    """
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot correlation
    plt.subplot(1, 2, 2)
    plt.plot(val_correlations, label='Val Correlation')
    plt.xlabel('Epoch')
    plt.ylabel('Spearman Correlation')
    plt.legend()
    plt.title('Validation Correlation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    plt.close()

def main(args):
    """
    Main function
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        h5_file=args.data_path,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        val_slide=args.val_slide,
        test_slide=args.test_slide
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    if args.model_type == 'single':
        model = CellMapModel(
            num_classes=args.num_classes,
            backbone=args.backbone,
            pretrained=True
        )
    elif args.model_type == 'multi_scale':
        model = MultiScaleCellMapModel(
            num_classes=args.num_classes,
            backbone=args.backbone,
            pretrained=True
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    model = model.to(device)
    
    # Define loss function
    criterion = CombinedLoss(
        mse_weight=args.mse_weight,
        spearman_weight=args.spearman_weight
    )
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Define scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Train model
    model, train_losses, val_losses, val_correlations = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir
    )
    
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CellMap model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='elucidata_ai_challenge_data.h5', help='Path to the h5 file')
    parser.add_argument('--patch_size', type=int, default=224, help='Size of the patches to extract')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--val_slide', type=str, default=None, help='Slide ID to use for validation')
    parser.add_argument('--test_slide', type=str, default='S_7', help='Slide ID to use for testing')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='single', choices=['single', 'multi_scale'], help='Model type')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'efficientnet_b0', 'densenet121'], help='Backbone architecture')
    parser.add_argument('--num_classes', type=int, default=35, help='Number of cell types to predict')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--mse_weight', type=float, default=0.7, help='Weight for MSE loss')
    parser.add_argument('--spearman_weight', type=float, default=0.3, help='Weight for Spearman loss')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    main(args)
