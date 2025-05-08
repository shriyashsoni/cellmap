import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import argparse
import h5py
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from dataset import HistologyDataset
from model import CellMapModel, MultiScaleCellMapModel
from torch.utils.data import DataLoader
from torchvision import transforms

def load_model(checkpoint_path, model_type='single', backbone='resnet50', num_classes=35, device='cuda'):
    """
    Load a trained model from a checkpoint
    """
    # Create model
    if model_type == 'single':
        model = CellMapModel(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=False
        )
    elif model_type == 'multi_scale':
        model = MultiScaleCellMapModel(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=False
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    return model

def predict(model, dataloader, device):
    """
    Make predictions with the model
    """
    model.eval()
    all_preds = []
    all_spot_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Get data
            images = batch['image'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Store predictions
            all_preds.append(outputs.cpu().numpy())
            
            # Store spot IDs (slide_id + coordinates)
            for i in range(len(batch['slide_id'])):
                slide_id = batch['slide_id'][i]
                coord = batch['coord'][i].numpy()
                spot_id = f"{slide_id}_{int(coord[0])}_{int(coord[1])}"
                all_spot_ids.append(spot_id)
    
    # Concatenate predictions
    all_preds = np.vstack(all_preds)
    
    return all_preds, all_spot_ids

def create_submission(predictions, spot_ids, output_path='submission.csv'):
    """
    Create a submission file
    """
    # Create dataframe
    df = pd.DataFrame(predictions, index=spot_ids)
    
    # Rename columns
    df.columns = [f'C{i+1}' for i in range(df.shape[1])]
    
    # Reset index
    df.index.name = 'ID'
    
    # Save to CSV
    df.to_csv(output_path)
    
    print(f"Submission saved to {output_path}")
    
    return df

def visualize_predictions(h5_file, slide_id, predictions, spot_ids, output_dir='visualizations'):
    """
    Visualize predictions as heatmaps overlaid on the histology image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load slide image
    with h5py.File(h5_file, 'r') as f:
        image = f['images'][slide_id][:]
    
    # Get spot coordinates and predictions for this slide
    slide_spots = []
    slide_preds = []
    
    for i, spot_id in enumerate(spot_ids):
        if spot_id.startswith(slide_id):
            # Extract coordinates from spot ID
            _, x, y = spot_id.split('_')
            x, y = int(x), int(y)
            
            slide_spots.append((x, y))
            slide_preds.append(predictions[i])
    
    # Convert to numpy arrays
    slide_spots = np.array(slide_spots)
    slide_preds = np.array(slide_preds)
    
    # Plot heatmaps for selected cell types
    num_cell_types = min(9, slide_preds.shape[1])
    
    plt.figure(figsize=(15, 15))
    
    # Plot original image
    plt.subplot(3, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot heatmaps for selected cell types
    for i in range(1, num_cell_types):
        plt.subplot(3, 3, i+1)
        plt.imshow(image, alpha=0.7)
        
        # Create scatter plot with color based on prediction value
        plt.scatter(
            slide_spots[:, 0],
            slide_spots[:, 1],
            c=slide_preds[:, i],
            cmap='viridis',
            alpha=0.7,
            s=50
        )
        
        plt.colorbar(label='Abundance')
        plt.title(f'Cell Type C{i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{slide_id}_heatmaps.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def main(args):
    """
    Main function
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint_path,
        model_type=args.model_type,
        backbone=args.backbone,
        num_classes=args.num_classes,
        device=device
    )
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    test_dataset = HistologyDataset(
        h5_file=args.data_path,
        slides=[args.test_slide],
        patch_size=args.patch_size,
        transform=transform,
        normalize_stain=True,
        mode='test'
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Make predictions
    predictions, spot_ids = predict(model, test_loader, device)
    
    # Create submission file
    submission_df = create_submission(
        predictions=predictions,
        spot_ids=spot_ids,
        output_path=args.output_path
    )
    
    # Visualize predictions
    if args.visualize:
        visualize_predictions(
            h5_file=args.data_path,
            slide_id=args.test_slide,
            predictions=predictions,
            spot_ids=spot_ids,
            output_dir=args.vis_dir
        )
    
    print("Inference complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference with CellMap model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='elucidata_ai_challenge_data.h5', help='Path to the h5 file')
    parser.add_argument('--patch_size', type=int, default=224, help='Size of the patches to extract')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--test_slide', type=str, default='S_7', help='Slide ID to use for testing')
    
    # Model parameters
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--model_type', type=str, default='single', choices=['single', 'multi_scale'], help='Model type')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'efficientnet_b0', 'densenet121'], help='Backbone architecture')
    parser.add_argument('--num_classes', type=int, default=35, help='Number of cell types to predict')
    
    # Output parameters
    parser.add_argument('--output_path', type=str, default='submission.csv', help='Path to save the submission file')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize predictions')
    parser.add_argument('--vis_dir', type=str, default='visualizations', help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    main(args)
