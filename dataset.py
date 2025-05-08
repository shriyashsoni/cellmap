import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class StainNormalization:
    """
    Macenko method for stain normalization
    """
    def __init__(self):
        self.HERef = np.array([[0.5626, 0.2159],
                              [0.7201, 0.8012],
                              [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])
        
    def normalize(self, img):
        """
        Normalize H&E stained image using Macenko method
        """
        img = img.astype(np.float32) / 255
        
        # Convert to OD space
        OD = -np.log(img + 1e-8)
        
        # Remove pixels with OD intensity less than β in any channel
        ODhat = OD[np.all(OD >= 0.15, axis=2), :]
        
        if ODhat.size == 0:
            return img * 255
        
        # Compute eigenvectors
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        
        # Extract two largest eigenvectors
        ind = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, ind]
        eigvals = eigvals[ind]
        
        # Project on the eigenvectors
        That = ODhat.dot(eigvecs[:, 0:2])
        
        # Get angle of each point
        phi = np.arctan2(That[:, 1], That[:, 0])
        
        # Find extremes (min and max angles)
        minPhi = np.percentile(phi, 1)
        maxPhi = np.percentile(phi, 99)
        
        # Calculate eigenvectors for hematoxylin and eosin
        vMin = eigvecs[:, 0:2].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:, 0:2].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
        
        # Project on the vectors
        if vMin[0] > vMax[0]:
            HE = np.array((vMin[:, 0], vMax[:, 0])).T
        else:
            HE = np.array((vMax[:, 0], vMin[:, 0])).T
        
        # Calculate concentrations
        Y = np.reshape(OD, (-1, 3))
        C = np.linalg.lstsq(HE, Y.T, rcond=None)[0]
        
        # Calculate maximal concentrations
        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
        
        # Normalize stains
        C = C / maxC[:, np.newaxis] * self.maxCRef[:, np.newaxis]
        
        # Recreate the image
        Inorm = np.exp(-self.HERef.dot(C))
        Inorm = np.reshape(Inorm.T, img.shape)
        
        # Convert back to RGB
        Inorm = np.clip(Inorm, 0, 1)
        Inorm = (Inorm * 255).astype(np.uint8)
        
        return Inorm

class HistologyDataset(Dataset):
    def __init__(self, h5_file, slides=None, patch_size=224, transform=None, normalize_stain=True, mode='train'):
        """
        Dataset for histology images and cell type abundances
        
        Args:
            h5_file (str): Path to the h5 file
            slides (list): List of slide IDs to use
            patch_size (int): Size of the patches to extract
            transform (callable): Optional transform to be applied on a sample
            normalize_stain (bool): Whether to apply stain normalization
            mode (str): 'train', 'val', or 'test'
        """
        self.h5_file = h5_file
        self.patch_size = patch_size
        self.transform = transform
        self.normalize_stain = normalize_stain
        self.mode = mode
        self.stain_normalizer = StainNormalization() if normalize_stain else None
        
        # Open the h5 file
        with h5py.File(h5_file, 'r') as f:
            # Get available slides
            available_slides = list(f['images'].keys())
            
            if slides is None:
                self.slides = available_slides
            else:
                self.slides = [s for s in slides if s in available_slides]
            
            # Get spot coordinates and cell type abundances
            self.spots = []
            self.abundances = []
            
            for slide_id in self.slides:
                # Get spot coordinates
                coords = f['spatial_coords'][slide_id][:]
                
                # Get cell type abundances if in train or val mode
                if mode != 'test':
                    abund = f['cell_abundance'][slide_id][:]
                else:
                    # For test mode, create dummy abundances
                    abund = np.zeros((len(coords), 35))
                
                # Store slide ID with each spot
                slide_ids = np.array([slide_id] * len(coords), dtype=object)
                
                # Add to lists
                for i in range(len(coords)):
                    self.spots.append((slide_ids[i], coords[i]))
                    self.abundances.append(abund[i])
            
            # Convert to numpy arrays
            self.abundances = np.array(self.abundances)
            
            # Store image shapes for each slide
            self.image_shapes = {}
            for slide_id in self.slides:
                self.image_shapes[slide_id] = f['images'][slide_id].shape
    
    def __len__(self):
        return len(self.spots)
    
    def __getitem__(self, idx):
        # Get spot info
        slide_id, coord = self.spots[idx]
        abundance = self.abundances[idx]
        
        # Extract patch
        with h5py.File(self.h5_file, 'r') as f:
            # Get image
            image = f['images'][slide_id][:]
            
            # Get patch coordinates
            x, y = int(coord[0]), int(coord[1])
            
            # Calculate patch boundaries
            half_size = self.patch_size // 2
            x_min = max(0, x - half_size)
            x_max = min(image.shape[1], x + half_size)
            y_min = max(0, y - half_size)
            y_max = min(image.shape[0], y + half_size)
            
            # Extract patch
            patch = image[y_min:y_max, x_min:x_max]
            
            # Pad if necessary
            if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                pad_width = [
                    (max(0, half_size - y), max(0, y + half_size - image.shape[0])),
                    (max(0, half_size - x), max(0, x + half_size - image.shape[1])),
                    (0, 0)
                ]
                patch = np.pad(patch, pad_width, mode='constant')
            
            # Apply stain normalization if enabled
            if self.normalize_stain:
                patch = self.stain_normalizer.normalize(patch)
            
            # Apply transformations if any
            if self.transform:
                patch = self.transform(patch)
            else:
                # Convert to tensor
                patch = torch.from_numpy(patch.transpose(2, 0, 1)).float() / 255.0
            
            # Return sample
            return {
                'image': patch,
                'abundance': torch.from_numpy(abundance).float(),
                'slide_id': slide_id,
                'coord': torch.tensor([x, y])
            }

def get_data_loaders(h5_file, patch_size=224, batch_size=32, val_slide=None, test_slide='S_7'):
    """
    Create train, validation, and test data loaders
    
    Args:
        h5_file (str): Path to the h5 file
        patch_size (int): Size of the patches to extract
        batch_size (int): Batch size
        val_slide (str): Slide ID to use for validation
        test_slide (str): Slide ID to use for testing
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get available slides
    with h5py.File(h5_file, 'r') as f:
        available_slides = list(f['images'].keys())
    
    # Set validation slide if not provided
    if val_slide is None and len(available_slides) > 1:
        val_slide = available_slides[-1]
    
    # Set train slides
    train_slides = [s for s in available_slides if s != val_slide and s != test_slide]
    
    # Create datasets
    train_dataset = HistologyDataset(
        h5_file=h5_file,
        slides=train_slides,
        patch_size=patch_size,
        transform=train_transform,
        normalize_stain=True,
        mode='train'
    )
    
    val_dataset = HistologyDataset(
        h5_file=h5_file,
        slides=[val_slide],
        patch_size=patch_size,
        transform=val_transform,
        normalize_stain=True,
        mode='val'
    )
    
    test_dataset = HistologyDataset(
        h5_file=h5_file,
        slides=[test_slide],
        patch_size=patch_size,
        transform=val_transform,
        normalize_stain=True,
        mode='test'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def visualize_sample(dataset, idx=0):
    """
    Visualize a sample from the dataset
    """
    sample = dataset[idx]
    
    # Convert tensor to numpy
    image = sample['image'].numpy().transpose(1, 2, 0)
    
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Slide: {sample['slide_id']}, Coord: {sample['coord'].numpy()}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(35), sample['abundance'].numpy())
    plt.title('Cell Type Abundances')
    plt.xlabel('Cell Type')
    plt.ylabel('Abundance')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create dataset
    dataset = HistologyDataset(
        h5_file='elucidata_ai_challenge_data.h5',
        patch_size=224,
        transform=None,
        normalize_stain=True
    )
    
    # Visualize a sample
    visualize_sample(dataset)
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        h5_file='elucidata_ai_challenge_data.h5',
        patch_size=224,
        batch_size=32
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
