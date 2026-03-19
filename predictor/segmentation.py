"""
Segmentation model for wing boundary detection.

This module provides the U-Net++ architecture enhanced with CoordConv layers
for precise wing boundary segmentation. The CoordConv augmentation helps the
model learn spatial structure by adding coordinate channels to the input.
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from config_loader import get_config

# Set default device
device = torch.device("cpu")

class AddCoords(nn.Module):
    """
    A layer that appends normalized x and y coordinate channels (and optionally a radial channel)
    to the input tensor for CoordConv operations.
    """
    def __init__(self, with_r=False):
        """
        Args:
            with_r (bool): Whether to include a radial distance channel sqrt(x^2 + y^2).
        """
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor (Tensor): Input of shape (B, C, H, W)
        
        Returns:
            Tensor: Output tensor with added coordinate channels.
        """
        batch_size, _, height, width = input_tensor.size()

        # Generate normalized x and y coordinates in range [-1, 1]
        xx_channel = torch.arange(width).repeat(1, height, 1)
        yy_channel = torch.arange(height).repeat(1, width, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (width - 1)
        yy_channel = yy_channel.float() / (height - 1)

        xx_channel = xx_channel * 2 - 1  # Normalize to [-1, 1]
        yy_channel = yy_channel * 2 - 1

        # Expand to batch size and add channel dimension
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).to(input_tensor.device)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).to(input_tensor.device)

        # Concatenate coordinates with input
        coords = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        # Optionally add radial distance channel
        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_channel ** 2)
            coords = torch.cat([coords, rr], dim=1)

        return coords


class CoordConvUnet(nn.Module):
    """
    A wrapper around a segmentation model that adds CoordConv channels to the input.
    """
    def __init__(self, base_model):
        """
        Args:
            base_model (nn.Module): A segmentation model (e.g., Unet++) with in_channels=3.
        """
        super().__init__()
        self.addcoords = AddCoords(with_r=False)
        self.base_model = base_model

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, 1, H, W) — original image.
        
        Returns:
            Tensor: Segmentation output from base model.
        """
        x = self.addcoords(x)  # Add x and y coord channels to input
        return self.base_model(x)


def get_model(family="mosquito", pretrained=True):
    """
    Instantiates the CoordConv-enhanced Unet++ segmentation model,
    loads pretrained weights, and returns it.

    Args:
        family (str): Insect family name
        pretrained (bool): Whether to load pretrained weights

    Returns:
        nn.Module: Fully constructed segmentation model.
    """
    config = get_config(family)
    model_path = config["model_paths"]["segmentation"]
    
    # Create the base Unet++ model with EfficientNet-b0 encoder
    base_model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,  # 1 channel image + 2 coord channels
        classes=1       # Binary segmentation output
    )

    # Wrap with CoordConv layer
    model = CoordConvUnet(base_model).to(device)
    #model = base_model.to(device)

    if pretrained:
        # Load pretrained weights
        model.load_state_dict(torch.load(model_path, map_location=device))

    return model
