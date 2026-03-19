"""
Landmark detection model for wing anatomical feature localization.

This module defines a custom CNN architecture for predicting anatomical
landmark positions on insect wings. The model uses both the original image
and segmentation mask as inputs to improve localization accuracy.

Architecture features:
- CoordConv layers for spatial awareness
- Multi-scale feature extraction
- Heatmap-based landmark prediction
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn

from config_loader import get_config

# Set default device
device = torch.device("cpu")

# ---------------------- Custom Layers ---------------------- #
class CoordConv(nn.Module):
    """
    CoordConv layer that appends normalized x and y coordinate channels
    to the input tensor before applying a standard 2D convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CoordConv, self).__init__()
        # Add 2 to in_channels to account for x and y coordinate channels
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        device = x.device

        # Generate x and y coordinate grids normalized to [-1, 1]
        yy_channel, xx_channel = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing="ij"
        )

        # Expand to batch size
        xx_channel = xx_channel.unsqueeze(0).expand(batch_size, 1, height, width)
        yy_channel = yy_channel.unsqueeze(0).expand(batch_size, 1, height, width)

        # Concatenate coordinate channels to input
        coord_channels = torch.cat([xx_channel, yy_channel], dim=1)
        x = torch.cat([x, coord_channels], dim=1)

        return self.conv(x)


class ConvBlock(nn.Module):
    """
    A basic convolutional block: Conv2d -> GroupNorm -> LeakyReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Hourglass(nn.Module):
    """
    Hourglass network for landmark localization, enhanced with CoordConv and segmentation map conditioning.
    """
    def __init__(self, in_channels, num_blocks=4, intermediate_channels=64, output_channels=17):
        """
        Args:
            in_channels (int): Number of input image channels (e.g., 1 for grayscale).
            num_blocks (int): Number of downsampling/upsampling blocks.
            intermediate_channels (int): Feature size in intermediate layers.
            output_channels (int): Number of output heatmap channels (landmarks).
        """
        super(Hourglass, self).__init__()
        self.num_blocks = num_blocks

        # Downsampling path
        self.coord_convs = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        for i in range(num_blocks):
            input_ch = in_channels if i == 0 else intermediate_channels
            # +1 to input for segmentation map
            self.coord_convs.append(CoordConv(input_ch + 1, intermediate_channels))
            self.down_blocks.append(ConvBlock(intermediate_channels, intermediate_channels))

        # Bottleneck block (lowest resolution)
        self.bottleneck = ConvBlock(intermediate_channels, intermediate_channels)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.up_blocks.append(ConvBlock(intermediate_channels, intermediate_channels))

        # Final output block to produce landmark heatmaps
        self.output_block = nn.Sequential(
            ConvBlock(intermediate_channels, intermediate_channels),
            nn.Conv2d(intermediate_channels, output_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, image, segmentation):
        """
        Forward pass for the hourglass model.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W).
            segmentation (Tensor): Segmentation map of shape (B, 1, H, W).

        Returns:
            Tensor: Output heatmaps for landmarks of shape (B, output_channels, H, W).
        """
        x = torch.cat([image, segmentation], dim=1)  # Combine inputs

        skip_connections = []

        # Downsampling with CoordConv + ConvBlock + MaxPool
        for i in range(self.num_blocks):
            x = self.coord_convs[i](x)
            x = self.down_blocks[i](x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

            # Resize segmentation to match new spatial size and concatenate again
            if i < self.num_blocks - 1:
                segmentation = F.interpolate(segmentation, size=x.shape[2:], mode='nearest')
                x = torch.cat([x, segmentation], dim=1)

        # Bottleneck processing
        x = self.bottleneck(x)

        # Upsampling with skip connections
        for i, up in enumerate(self.up_blocks):
            # Upsample and add skip connection
            x = F.interpolate(x, size=skip_connections[-(i + 1)].shape[2:], mode='nearest')
            x = x + skip_connections[-(i + 1)]
            x = up(x)

        return self.output_block(x)


def get_model(family="mosquito"):
    """
    Instantiates the Hourglass model, loads pretrained weights, and returns it.

    Args:
        family (str): Insect family name

    Returns:
        Hourglass: The landmark detection model with loaded weights.
    """
    config = get_config(family)
    N_landmarks = config["N_landmarks"]
    model_path = config["model_paths"]["landmark"]
    HG_blocks = 4  # Number of hourglass levels

    # Initialize model
    landmark_model = Hourglass(
        in_channels=1,
        num_blocks=HG_blocks,
        intermediate_channels=64,
        output_channels=N_landmarks
    )

    # Load pretrained weights
    landmark_model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    return landmark_model

def load_partial_weights(model, model_path, ignored_keys=("output_block.1.weight", "output_block.1.bias")):
    """
    Loads pretrained weights into a model, skipping layers with mismatched shapes.

    Args:
        model (nn.Module): Model to load weights into.
        checkpoint_path (str): Path to the .pt or .pth file with weights.
        ignored_keys (tuple): Keys to ignore (e.g., final output layer).
    """
    state_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()

    # Filter out keys to ignore
    filtered_dict = {
        k: v for k, v in state_dict.items() if k in model_dict and k not in ignored_keys
    }

    # Update model state dict
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(filtered_dict)} parameters from pretrained model (skipped output layer)")

def get_pretrained_model(family="mosquito"):
    """
    Get a pretrained model with partial weight loading.

    Args:
        family (str): Insect family name

    Returns:
        Hourglass: The landmark detection model with loaded weights
    """
    config = get_config(family)
    N_landmarks = config["N_landmarks"]

    landmark_model = Hourglass(
        in_channels=1,
        num_blocks=4,
        intermediate_channels=64,
        output_channels=N_landmarks
    )

    # Load all weights except the final layer
    load_partial_weights(landmark_model, config["model_paths"]["landmark"])

    return landmark_model


