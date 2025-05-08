import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
from tqdm import tqdm

from dataset import HistologyDataset
from model import CellMapModel

def visualize_dataset(h5_file, slide_id, output_dir='visualizations'):
    """
    Visualize a slide and its spots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load slide image and spot coordinates
    with h5py.File(h5_file, 'r') as f:
        image = f['images'][slide_id][:]
        coords = f['spatial_coords'][slide_id][:]
        
        # Load cell abundances if available
        if slide_id in f['cell_abundance']:
            abundances = f['cell_abundance'][slide_id][:]
        else:
            abundances = None
    
    # Plot slide with spots
    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=10, alpha=0.5)
    plt.title(f'Slide {slide_id} with Spots')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'{slide_id}_spots.png'))
    plt.close()
    
    # If abundances are available, plot heatmaps for selected cell types
    if abundances is not None:
        num_cell_types = min(9, abundances.shape[1])
        
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
            
            # Create scatter plot with color based on abundance value
            plt.scatter(
                coords[:, 0],
                coords[:, 1],
                c=abundances[:, i],
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

def analyze_cell_type_distribution(h5_file, output_dir='visualizations'):
    """
    Analyze the distribution of cell types across slides
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load cell abundances for all slides
    with h5py.File(h5_file, 'r') as f:
        slides = list(f['cell_abundance'].keys())
        
        # Collect abundances
        all_abundances = []
        slide_labels = []
        
        for slide_id in slides:
            abundances = f['cell_abundance'][slide_id][:]
            all_abundances.append(abundances)
            slide_labels.extend([slide_id] * len(abundances))
        
        # Concatenate abundances
        all_abundances = np.vstack(all_abundances)
    
    # Create dataframe
    df = pd.DataFrame(all_abundances, columns=[f'C{i+1}' for i in range(all_abundances.shape[1])])
    df['Slide'] = slide_labels
    
    # Plot distribution of cell types
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df.melt(id_vars='Slide', var_name='Cell Type', value_name='Abundance'))
    plt.xticks(rotation=90)
    plt.title('Distribution of Cell Type Abundances Across Slides')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cell_type_distribution.png'))
    plt.close()
    
    # Plot correlation between cell types
    plt.figure(figsize=(12, 10))
    corr = df.drop('Slide', axis=1).corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Cell Types')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cell_type_correlation.png'))
    plt.close()
    
    # Dimensionality reduction for visualization
    X = all_abundances
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    for slide_id in slides:
        mask = np.array(slide_labels) == slide_id
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=slide_id, alpha=0.7)
    
    plt.title('PCA of Cell Type Abundances')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'pca_visualization.png'))
    plt.close()
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    for slide_id in slides:
        mask = np.array(slide_labels) == slide_id
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=slide_id, alpha=0.7)
    
    plt.title('t-SNE of Cell Type Abundances')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))
    plt.close()
    
    print(f"Analysis saved to {output_dir}")

def extract_features(model, dataloader, device):
    """
    Extract features from the model
    """
    model.eval()
    all_features = []
    all_spot_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # Get data
            images = batch['image'].to(device)
            
            # Forward pass (extract features before the final layer)
            features = model.backbone(images)
            
            # Store features
            all_features.append(features.cpu().numpy())
            
            # Store spot IDs (slide_id + coordinates)
            for i in range(len(batch['slide_id'])):
                slide_id = batch['slide_id'][i]
                coord = batch['coord'][i].numpy()
                spot_id = f"{slide_id}_{int(coord[0])}_{int(coord[1])}"
                all_spot_ids.append(spot_id)
    
    # Concatenate features
    all_features = np.vstack(all_features)
    
    return all_features, all_spot_ids

def visualize_feature_space(model, h5_file, slides, output_dir='visualizations', patch_size=224):
    """
    Visualize the feature space of the model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = HistologyDataset(
        h5_file=h5_file,
        slides=slides,
        patch_size=patch_size,
        transform=transform,
        normalize_stain=True,
        mode='train'
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Extract features
    features, spot_ids = extract_features(model, dataloader, device)
    
    # Get slide IDs from spot IDs
    slide_labels = [spot_id.split('_')[0] for spot_id in spot_ids]
    
    # Dimensionality reduction for visualization
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    for slide_id in slides:
        mask = np.array(slide_labels) == slide_id
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=slide_id, alpha=0.7)
    
    plt.title('PCA of Model Features')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'feature_pca_visualization.png'))
    plt.close()
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    for slide_id in slides:
        mask = np.array(slide_labels) == slide_id
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=slide_id, alpha=0.7)
    
    plt.title('t-SNE of Model Features')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'feature_tsne_visualization.png'))
    plt.close()
    
    print(f"Feature visualizations saved to {output_dir}")

def generate_grad_cam(model, image, target_layer_name='layer4'):
    """
    Generate Grad-CAM visualization for a single image
    """
    # Set model to evaluation mode
    model.eval()
    
    # Convert image to tensor
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        image = image.unsqueeze(0)
    
    # Normalize image
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalize(image)
    
    # Move to device
    device = next(model.parameters()).device
    image = image.to(device)
    
    # Get target layer
    target_layer = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break
    
    if target_layer is None:
        raise ValueError(f"Target layer {target_layer_name} not found in model")
    
    # Register hooks
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    output = model(image)
    
    # Get the most important class (highest prediction)
    pred_class = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()
    
    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()
    
    # Get activations and gradients
    activation = activations[0].detach()
    gradient = gradients[0].detach()
    
    # Calculate weights
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    
    # Generate CAM
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    
    # Normalize CAM
    cam = F.interpolate(cam, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    # Convert to numpy
    cam = cam.squeeze().cpu().numpy()
    
    return cam, pred_class

def visualize_grad_cam(model, h5_file, slide_id, spot_idx=0, output_dir='visualizations', patch_size=224):
    """
    Visualize Grad-CAM for a specific spot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load slide image and spot coordinates
    with h5py.File(h5_file, 'r') as f:
        image = f['images'][slide_id][:]
        coords = f['spatial_coords'][slide_id][:]
        
        # Get spot coordinates
        x, y = int(coords[spot_idx, 0]), int(coords[spot_idx, 1])
        
        # Extract patch
        half_size = patch_size // 2
        x_min = max(0, x - half_size)
        x_max = min(image.shape[1], x + half_size)
        y_min = max(0, y - half_size)
        y_max = min(image.shape[0], y + half_size)
        
        patch = image[y_min:y_max, x_min:x_max]
        
        # Pad if necessary
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            pad_width = [
                (max(0, half_size - y), max(0, y + half_size - image.shape[0])),
                (max(0, half_size - x), max(0, x + half_size - image.shape[1])),
                (0, 0)
            ]
            patch = np.pad(patch, pad_width, mode='constant')
    
    # Generate Grad-CAM
    cam, pred_class = generate_grad_cam(model, patch)
    
    # Visualize
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(patch)
    plt.title('Original Patch')
    plt.axis('off')
    
    # Grad-CAM
    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title(f'Grad-CAM (Class C{pred_class+1})')
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(patch)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{slide_id}_spot_{spot_idx}_gradcam.png'))
    plt.close()
    
    print(f"Grad-CAM visualization saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    # Visualize dataset
    visualize_dataset(
        h5_file='elucidata_ai_challenge_data.h5',
        slide_id='S_1',
        output_dir='visualizations'
    )
    
    # Analyze cell type distribution
    analyze_cell_type_distribution(
        h5_file='elucidata_ai_challenge_data.h5',
        output_dir='visualizations'
    )
