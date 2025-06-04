# HEBlock + HENet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class HEBlock(nn.Module):
    """
    Optimized HE (Hide and Enhance) Block implementation.
    Vectorized implementation to eliminate slow Python loops.
    """
    def __init__(self, beta=0.5):
        """
        Args:
            beta: weight of mask (default: 0.5 as recommended in the paper)
        """
        super(HEBlock, self).__init__()
        self.beta = beta

    def forward(self, x):
        """
        Args:
            x: input feature map of shape (batch_size, C, H, W)
        Returns:
            Modified feature map with suppressed maximum activations
        """
        if not self.training:  # Only apply during training
            return x
        
        # Get shape information
        batch_size, channels, h, w = x.size()
        
        # Find maximum values for each channel in each sample in batch
        # Shape: [batch_size, channels, 1, 1]
        max_vals = x.view(batch_size, channels, -1).max(dim=2)[0].view(batch_size, channels, 1, 1)
        
        # Create masks where the value equals the max value
        # Broadcasting handles the comparison efficiently
        mask = (x == max_vals).float()
        
        # Apply the beta factor to maximum values using the mask
        # This is a vectorized operation that replaces the nested loops
        output = torch.where(mask == 1, self.beta * x, x)
        
        return output


class HENet(nn.Module):
    """
    Optimized HENet implementation for font recognition.
    """
    def __init__(self, num_classes=200, beta=0.5, use_amp=True):
        """
        Args:
            num_classes: Number of font classes (default: 2383)
            beta: Weight for the HE Block mask (default: 0.5)
            use_amp: Whether to use Automatic Mixed Precision (default: True)
        """
        super(HENet, self).__init__()
        
        # Track whether to use mixed precision
        self.use_amp = use_amp
        
        # Load pretrained ResNet18 - efficient backbone architecture
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the first convolutional layer to accept grayscale input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final fully connected layer to use as feature extractor
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # 1x1 convolution to match the number of classes (more efficient than FC for large number of classes)
        self.conv_final = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # Optimized HE Block
        self.he_block = HEBlock(beta=beta)
        
        # Global average pooling for efficient dimensionality reduction
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        # Use AMP if specified (faster computation with minimal accuracy loss)
        with torch.cuda.amp.autocast() if self.use_amp and torch.cuda.is_available() else torch.no_grad():
            # Feature extraction using ResNet backbone
            x = self.features(x)
            
            # 1x1 convolution to get class-specific activation maps
            x = self.conv_final(x)
            
            # Apply HE Block during training (now optimized)
            x = self.he_block(x)
            
            # Global average pooling and flatten
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            
        return x