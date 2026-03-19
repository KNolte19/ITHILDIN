"""
Classification model for species identification.

This module provides functionality to load a pretrained classification model
for identifying mosquito/insect species from wing images.
"""

import torch

from config_loader import get_config

# Set default device to CPU (can be overridden in CONFIG)
device = torch.device("cpu")


def get_model(family="mosquito"):
    """
    Load pretrained classification model for species identification.

    The model is loaded from the path specified in CONFIG and mapped to
    the configured device (CPU, CUDA, or MPS).

    Args:
        family (str): Insect family name

    Returns:
        torch.nn.Module: Loaded PyTorch model ready for inference

    Warning:
        Uses weights_only=False which can execute arbitrary code from untrusted
        model files. Only use with trusted model sources. This is necessary
        because the model architecture is saved along with the weights.
    """
    config = get_config(family)
    model = torch.load(
        config["model_paths"]["classification"],
        weights_only=False,
        map_location=device,
    )
    return model
