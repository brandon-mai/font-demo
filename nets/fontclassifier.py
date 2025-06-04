import torch.nn as nn
import torch

# FontClassifier initialization
class FontClassifier(nn.Module):
    def __init__(self, pretrained_scae, num_classes=200, normalization_type="batch_norm", 
                 use_dropout=False, dropout_prob=0.3, activation="relu"):
        super().__init__()
        self.pretrained_scae = pretrained_scae  # Use pretrained SCAE encoder
        
        # Define helper functions for creating layers
        def norm_layer(num_features, spatial_size=None):
            if normalization_type == "batch_norm":
                return nn.BatchNorm2d(num_features)
            elif normalization_type == "group_norm":
                return nn.GroupNorm(num_groups=8, num_channels=num_features)
            elif normalization_type == "layer_norm" and spatial_size is not None:
                return nn.LayerNorm([num_features, spatial_size, spatial_size])
            else:
                return nn.Identity()

        def activation_layer():
            return nn.LeakyReLU(inplace=True) if activation == "leaky_relu" else nn.ReLU(inplace=True)

        def dropout_layer():
            return nn.Dropout2d(dropout_prob) if use_dropout else nn.Identity()
        
        # CNN head after the SCAE encoder
        # SCAE encoder output is 128 x 26 x 26
        self.cnn_head = nn.Sequential(
            # Conv layer 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Out: 256 x 12 x 12
            norm_layer(256, 12),
            activation_layer(),
            
            # Conv layer 5
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Out: 256 x 12 x 12
            norm_layer(256, 13),
            activation_layer(),
            dropout_layer()
        )
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        # Input size is 256 * 12 * 12 = 43,264
        self.fully_connected = nn.Sequential(
            nn.Linear(256 * 12 * 12, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob if use_dropout else 0),
            
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob if use_dropout else 0),
            
            nn.Linear(2048, num_classes),
            # nn.Softmax(dim=1) no softmax here bro, crossentropy does the softmax automatically
        )

    def forward(self, x):
        # Use the encoder part of SCAE
        x = self.pretrained_scae.encoder(x)
        # Continue with additional CNN layers
        x = self.cnn_head(x)
        
        # Flatten and apply fully connected layers
        x = self.flatten(x)
        x = self.fully_connected(x)
        
        return x
