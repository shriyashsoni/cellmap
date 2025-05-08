import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CellMapModel(nn.Module):
    def __init__(self, num_classes=35, backbone='resnet50', pretrained=True):
        """
        CellMap model for predicting cell type abundances from histology images
        
        Args:
            num_classes (int): Number of cell types to predict
            backbone (str): Backbone architecture to use
            pretrained (bool): Whether to use pretrained weights
        """
        super(CellMapModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Initialize backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'densenet121':
            self.backbone = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
            nn.ReLU()  # Ensure non-negative outputs for abundance values
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Predict abundances
        abundances = self.regression_head(features)
        
        return abundances

class MultiScaleCellMapModel(nn.Module):
    def __init__(self, num_classes=35, backbone='resnet50', pretrained=True):
        """
        Multi-scale CellMap model for predicting cell type abundances from histology images
        
        Args:
            num_classes (int): Number of cell types to predict
            backbone (str): Backbone architecture to use
            pretrained (bool): Whether to use pretrained weights
        """
        super(MultiScaleCellMapModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Initialize backbones for different scales
        if backbone == 'resnet50':
            self.backbone1 = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim1 = self.backbone1.fc.in_features
            self.backbone1.fc = nn.Identity()
            
            self.backbone2 = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim2 = self.backbone2.fc.in_features
            self.backbone2.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            self.backbone1 = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim1 = self.backbone1.classifier[1].in_features
            self.backbone1.classifier = nn.Identity()
            
            self.backbone2 = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim2 = self.backbone2.classifier[1].in_features
            self.backbone2.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim1 + self.feature_dim2, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
            nn.ReLU()  # Ensure non-negative outputs for abundance values
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x1, x2):
        # Extract features from different scales
        features1 = self.backbone1(x1)
        features2 = self.backbone2(x2)
        
        # Concatenate features
        features = torch.cat([features1, features2], dim=1)
        
        # Fuse features
        fused = self.fusion(features)
        
        # Predict abundances
        abundances = self.regression_head(fused)
        
        return abundances

class EnsembleModel(nn.Module):
    def __init__(self, models):
        """
        Ensemble of CellMap models
        
        Args:
            models (list): List of CellMap models
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        # Get predictions from all models
        predictions = [model(x) for model in self.models]
        
        # Average predictions
        ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        
        return ensemble_pred

# Custom loss functions
class SpearmanCorrelationLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(SpearmanCorrelationLoss, self).__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        # Reshape if needed
        if pred.dim() == 3:
            pred = pred.view(-1, pred.size(2))
        if target.dim() == 3:
            target = target.view(-1, target.size(2))
        
        # Calculate ranks
        pred_ranks = self._rank(pred)
        target_ranks = self._rank(target)
        
        # Calculate Spearman correlation
        pred_ranks_centered = pred_ranks - pred_ranks.mean(dim=0, keepdim=True)
        target_ranks_centered = target_ranks - target_ranks.mean(dim=0, keepdim=True)
        
        covariance = (pred_ranks_centered * target_ranks_centered).mean(dim=0)
        pred_std = torch.sqrt(torch.var(pred_ranks, dim=0, unbiased=False) + self.eps)
        target_std = torch.sqrt(torch.var(target_ranks, dim=0, unbiased=False) + self.eps)
        
        correlation = covariance / (pred_std * target_std + self.eps)
        
        # Return negative mean correlation (to minimize)
        return -correlation.mean()
    
    def _rank(self, x):
        # Get the ranks of x
        batch_size, dim = x.shape
        ranks = torch.zeros_like(x)
        
        for i in range(dim):
            ranks[:, i] = self._rank_column(x[:, i])
        
        return ranks
    
    def _rank_column(self, x):
        # Get the ranks of a column
        sorted_indices = torch.argsort(x)
        ranks = torch.zeros_like(x)
        
        for i, idx in enumerate(sorted_indices):
            ranks[idx] = i + 1
        
        return ranks

class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.7, spearman_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.spearman_loss = SpearmanCorrelationLoss()
        self.mse_weight = mse_weight
        self.spearman_weight = spearman_weight
    
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        spearman = self.spearman_loss(pred, target)
        
        return self.mse_weight * mse + self.spearman_weight * spearman

# Example usage
if __name__ == "__main__":
    # Create model
    model = CellMapModel(num_classes=35, backbone='resnet50', pretrained=True)
    
    # Print model summary
    print(model)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
