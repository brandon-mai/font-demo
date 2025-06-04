import torch.nn as nn
import torch


class AlexNetClassifier(nn.Module):
    def __init__(self, num_classes=200):
        super(AlexNetClassifier, self).__init__()
        
        # Feature extraction layers (convolutional layers)
        self.features = nn.Sequential(
            # First conv layer: input 1x105x105 -> output 96x25x25
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Second conv layer: 96x25x25 -> 256x12x12
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Third conv layer: 256x12x12 -> 384x12x12
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fourth conv layer: 384x12x12 -> 384x12x12
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fifth conv layer: 384x12x12 -> 256x6x6
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive pooling to ensure consistent output size
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classifier layers (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        # Pass through feature extraction layers
        x = self.features(x)
        
        # Apply adaptive pooling
        x = self.avgpool(x)
        
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)
        
        # Pass through classifier
        x = self.classifier(x)
        
        return x