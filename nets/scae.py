# SCAE Initialization
import torch.nn as nn


class SCAE(nn.Module):
    def __init__(self, normalization_type="batch_norm", use_dropout=False, dropout_prob=0.3, activation="relu"):
        super(SCAE, self).__init__()

        def norm_layer(num_features):
            if normalization_type == "batch_norm":
                return nn.BatchNorm2d(num_features)
            elif normalization_type == "group_norm":
                return nn.GroupNorm(num_groups=8, num_channels=num_features)
            elif normalization_type == "layer_norm":
                return nn.LayerNorm([num_features, 12, 12])  # Updated for 12x12 feature maps
            else:
                return nn.Identity()

        def activation_layer():
            return nn.LeakyReLU(inplace=True) if activation == "leaky_relu" else nn.ReLU(inplace=True)

        def dropout_layer():
            return nn.Dropout2d(dropout_prob) if use_dropout else nn.Identity()

        # Encoder: Input 105x105 -> Output 12x12
        self.encoder = nn.Sequential(
            # Layer 1: 105x105 -> 48x48
            nn.Conv2d(1, 64, kernel_size=11, stride=2, padding=0),
            norm_layer(64),
            activation_layer(),
            dropout_layer(),
            
            # Layer 2: 48x48 -> 24x24
            nn.MaxPool2d(2, 2),
            
            # Layer 3: 24x24 -> 24x24
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            norm_layer(128),
            activation_layer(),
            dropout_layer(),
            
            # Layer 4: 24x24 -> 12x12 (added to get 12x12 output)
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder: Input 12x12 -> Output 105x105
        self.decoder = nn.Sequential(
            # Layer 1: 12x12 -> 24x24
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            norm_layer(64),
            activation_layer(),
            dropout_layer(),
            
            # Layer 2: 24x24 -> 48x48
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            norm_layer(32),
            activation_layer(),
            dropout_layer(),
            
            # Layer 3: 48x48 -> 105x105 (with precise output size)
            nn.ConvTranspose2d(32, 1, kernel_size=14, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Pass through encoder
        if x.size(1) == 3:
            # Use standard RGB to grayscale conversion: 0.299*R + 0.587*G + 0.114*B
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        for layer in self.encoder:
            x = layer(x)
            # print(x.shape)

        for layer in self.decoder:
            x = layer(x)
            # print(x.shape)
            
        return x