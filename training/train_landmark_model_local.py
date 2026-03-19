"""
ITHILDIN Landmark Model Training Script (Local Version)

This script trains a landmark detection model for wing analysis using a Hourglass network architecture.
It's designed to run locally with direct access to the training data in the repository.

For Colab version with Google Drive integration, use train_landmark_model_colab.ipynb instead.

Author: ITHILDIN Project
"""

import os
import sys
import torch
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage as ski
import albumentations as A
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# ============================================================================
# CONFIGURATION PARAMETERS - Modify these for your experiment
# ============================================================================

# Experiment Settings
EXPERIMENT = "data_droso"  # Experiment name (should match data directory)
DATA_PATH = "./training/data_droso/"  # Path to training data directory (relative to repo root)

# Model Architecture Parameters
N_LANDMARKS = 15  # Number of landmarks to detect
HG_BLOCKS = 4  # Number of Hourglass blocks

# Training Hyperparameters
LEARNING_RATE = 5e-4  # Initial learning rate
EPOCHS = 2  # Number of training epochs
BATCH_SIZE = 4  # Batch size for training
IMAGE_HEIGHT, IMAGE_WIDTH = 240, 480  # Input image dimensions

# Loss Function Parameters
ALPHA = 0.5  # Weight balance between different loss components

# Learning Rate Scheduler Parameters
WARMUP_EPOCHS = None  # Will be auto-calculated as 25% of total epochs if None
WARMUP_FACTOR = 0.1  # Starting learning rate multiplier for warmup

# Cross-Validation Settings
K_FOLDS = 5  # Number of folds for cross-validation
FOLD = 4  # Which fold to train on (0 to K_FOLDS-1)

# Model Paths
PRETRAINED_MODEL_PATH = None  # Path to pretrained segmentation model weights (None to skip)
OUTPUT_DIR = "./training/data_droso/"  # Directory to save trained models

# Hardware Settings
DEVICE = None  # Will auto-detect if None (cuda/mps/cpu)

# Visualization Settings
SHOW_PREVIEW = False  # Set to True to display training data preview
SAVE_PREVIEW = True  # Set to True to save preview images

# ============================================================================
# SETUP
# ============================================================================

# Auto-calculate warmup epochs if not specified
if WARMUP_EPOCHS is None:
    WARMUP_EPOCHS = np.ceil(EPOCHS * 0.25)

# Auto-detect device if not specified
if DEVICE is None:
    DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
print(f"Using {DEVICE} device")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"ITHILDIN LANDMARK MODEL TRAINING")
print(f"{'='*60}")
print(f"Experiment: {EXPERIMENT}")
print(f"Data Path: {DATA_PATH}")
print(f"N Landmarks: {N_LANDMARKS}")
print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Fold: {FOLD}/{K_FOLDS}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"{'='*60}\n")

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading training data...")

# Construct data paths
data_prefix = os.path.join(DATA_PATH, f"forlandmark_{EXPERIMENT}")
file_list = np.load(f"{data_prefix}_images.npy")
heatmap_list = np.load(f"{data_prefix}_heatmap.npy")
path_list = np.load(f"{data_prefix}_paths.npy")

print(f"Loaded {len(file_list)} images")
print(f"Image shape: {file_list.shape}")
print(f"Heatmap shape: {heatmap_list.shape}")

# ============================================================================
# SPLIT DATA INTO FOLDS
# ============================================================================

print(f"\nSplitting data into {K_FOLDS} folds...")

# Create KFold splitter
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# Generate all fold splits
splits = list(kf.split(np.arange(len(file_list))))

# Select the specified fold
train_idx, test_idx = splits[FOLD]

train_file_list = file_list[train_idx]
test_file_list = file_list[test_idx]

train_label_list = heatmap_list[train_idx]
test_label_list = heatmap_list[test_idx]

train_path_list = path_list[train_idx]
test_path_list = path_list[test_idx]

print(f"Training samples: {len(train_file_list)}")
print(f"Testing samples: {len(test_file_list)}")

# ============================================================================
# DATASET CLASS
# ============================================================================

class CustomDataset(Dataset):
    """
    Custom Dataset for landmark detection training.
    
    Expected inputs:
    - file_image: array-like of shape (N, H, W, C) with values in [0, 255]
    - file_heatmap: array-like of shape (N, H, W, K) with heatmap values
    - file_path: array-like of shape (N,) with image paths
    - augment_transforms: torchvision transforms for augmentation
    - augment_bool: whether to apply augmentations
    """
    def __init__(self, file_image, file_heatmap, file_path, augment_transforms, augment_bool=False):
        self.input_img = file_image
        self.heatmap = file_heatmap
        self.path = file_path
        self.augment_transforms = augment_transforms
        self.augment_bool = augment_bool

    def __len__(self):
        return len(self.input_img)

    def minmax_scale_per_layer(self, tensor):
        """Scale each channel independently to [0, 1]."""
        C = tensor.shape[0]
        scaled = torch.zeros_like(tensor)
        for c in range(C):
            channel = tensor[c]
            min_val = channel.min()
            max_val = channel.max()
            if max_val - min_val > 1e-8:
                scaled[c] = (channel - min_val) / (max_val - min_val)
            else:
                scaled[c] = channel
        return scaled

    def CLAHE_transform(self, image_tensor):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        # Reduce dimension
        image = torch.mean(image_tensor, dim=0).numpy()
        
        # Apply CLAHE
        clahe_image = ski.exposure.equalize_adapthist(image, clip_limit=.1, nbins=128)

        # Use median filter to reduce noise
        clahe_image = ski.filters.median(clahe_image, ski.morphology.disk(2))

        return torch.tensor(clahe_image, dtype=torch.float32)

    def geo_transforms(self, image: torch.Tensor, heatmap: torch.Tensor, mask: torch.Tensor = None):
        # Random rotation angle between -5 and 5 degrees
        angle = float(torch.empty(1).uniform_(-5, 5).item())

        # Rotate image and heatmap with bilinear interpolation
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, expand=False)
        heatmap = TF.rotate(heatmap, angle, interpolation=TF.InterpolationMode.BILINEAR, expand=False)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, expand=False)

        # Random horizontal flip with 50% prob for all tensors
        if torch.rand(1).item() > 0.5:
            image = TF.hflip(image)
            heatmap = TF.hflip(heatmap)
            mask = TF.hflip(mask)

        return image, heatmap, mask

    def __getitem__(self, idx):
        # Load Data from numpy-like arrays and convert immediately to torch tensors in C,H,W layout
        # Assumes stored arrays are (H, W, C)
        img_np = self.input_img[idx].astype(np.float32)
        heat_np = self.heatmap[idx].astype(np.float32)
        path = self.path[idx]

        # Use floating point torch tensors, channel-first
        image = torch.from_numpy(img_np).permute(2, 0, 1).float().div(255.0)  # (C, H, W)
        heatmap = torch.from_numpy(heat_np).permute(2, 0, 1).float().div(255.0)  # (K, H, W)

        # Foreground mask determined from first image channel
        mask = (image[0] != 0)  # (H, W) boolean
        mask_tensor = mask.float().unsqueeze(0)  # (1, H, W)

        # Augment Data (geometric + optional extra augment)
        if self.augment_bool:
            # Apply the same geometric transform to image, heatmap, and mask
            image, heatmap, mask_tensor = self.geo_transforms(image, heatmap, mask_tensor)
            image = self.augment_transforms(image)

        # mask_tensor is (1, H, W) float -> threshold/round to boolean
        mask = mask_tensor.squeeze(0).round().bool()  # (H, W)

        # Generate Loss weights: sum spatially across all heatmap channels, then repeat per-channel
        summed = heatmap.sum(dim=0, keepdim=False)  # (H, W)
        weight = summed.unsqueeze(0).repeat(heatmap.shape[0], 1, 1)  # (K, H, W)

        # Smooth Landmarks and weights with gaussian blur
        # gaussian_blur accepts (C,H,W) tensors
        heatmap = TF.gaussian_blur(heatmap, kernel_size=(5, 5), sigma=(13.0,))
        weight = TF.gaussian_blur(weight, kernel_size=(5, 5), sigma=(13.0,))

        # Grayscale + CLAHE transform (returns (H, W))
        image = torch.clamp(image, min=0.0, max=1.0)
        image_gray = self.CLAHE_transform(image)

        # set all values outside the mask to 0
        image_gray[~mask] = 0.0

        # add channel dimension back -> (1, H, W)
        image_out = image_gray.unsqueeze(0)

        # Normalize heatmap and weight per-channel
        heatmap = self.minmax_scale_per_layer(heatmap)
        weight = self.minmax_scale_per_layer(weight)

        # Final clamps
        heatmap = torch.clamp(heatmap, min=0.0, max=1.0)
        weight = torch.clamp(weight, min=0.0, max=1.0)

        return (image_out, heatmap, weight, path)

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

augment_transforms = transforms.Compose([
    v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    v2.GaussianBlur(kernel_size=3, sigma=(0.75, 1.25)),
    v2.RandomPosterize(bits=8, p=0.1),
    v2.RandomAutocontrast(p=0.2),
    v2.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
    v2.RandomGrayscale(p=0.2),
])

segment_trans = A.Compose([
    A.PixelDropout(dropout_prob=0.1, p=0.5),
    A.CoarseDropout(p=1, num_holes_range=(3, 15),hole_height_range=(5, 25), hole_width_range=(5, 25)),
    A.Morphological(scale=(1, 3), operation="dilation", p=0.5),
    A.Morphological(scale=(1, 3), operation="erosion", p=0.5),
    A.MedianBlur((3, 5), p=0.75),
])

# ============================================================================
# CREATE DATASETS AND DATALOADERS
# ============================================================================

print("\nCreating datasets and dataloaders...")

# Create datasets
train_dataset = CustomDataset(
    file_image=train_file_list,
    file_heatmap=train_label_list,
    file_path=train_path_list,
    augment_transforms=augment_transforms,
    augment_bool=True
)

test_dataset = CustomDataset(
    file_image=test_file_list,
    file_heatmap=test_label_list,
    file_path=test_path_list,
    augment_transforms=None,
    augment_bool=False
)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False)

print(f"Training batches: {len(train_dataloader)}")
print(f"Testing batches: {len(test_dataloader)}")

# ============================================================================
# PREVIEW DATA (Optional)
# ============================================================================

if SHOW_PREVIEW or SAVE_PREVIEW:
    print("\nGenerating data preview...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    ax = axes.ravel()

    image_batch, heatmap_batch, weight_batch, path_batch = next(iter(train_dataloader))

    for i in range(4):
        image = image_batch.numpy()[i][0]
        image = ski.transform.resize(image, (240, 480), anti_aliasing=False, order=1)
        heatmap = np.sum(heatmap_batch.numpy()[i], axis=0)

        ax[i].imshow(image, cmap='gray')
        ax[i].imshow(heatmap, cmap='Blues', alpha=0.5)
        ax[i].axis('off')

    plt.tight_layout()
    
    if SAVE_PREVIEW:
        preview_path = os.path.join(OUTPUT_DIR, f"landmark_training_preview_fold{FOLD}.png")
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        print(f"Preview saved to: {preview_path}")
    
    if SHOW_PREVIEW:
        plt.show()
    else:
        plt.close()

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

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
    """A basic convolutional block: Conv2d -> GroupNorm -> LeakyReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Hourglass(nn.Module):
    """
    Hourglass network for landmark localization,
    enhanced with CoordConv and segmentation map conditioning.
    """
    def __init__(self, in_channels, num_blocks=4, intermediate_channels=64, output_channels=N_LANDMARKS):
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

        # Bottleneck block
        self.bottleneck = ConvBlock(intermediate_channels, intermediate_channels)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.up_blocks.append(ConvBlock(intermediate_channels, intermediate_channels))

        # Final output block
        self.output_block = nn.Sequential(
            ConvBlock(intermediate_channels, intermediate_channels),
            nn.Conv2d(intermediate_channels, output_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, image, segmentation):
        """
        Forward pass for the hourglass model.
        
        Args:
            image: Input image tensor of shape (B, C, H, W)
            segmentation: Segmentation map of shape (B, 1, H, W)
            
        Returns:
            Output heatmaps for landmarks of shape (B, output_channels, H, W)
        """
        x = torch.cat([image, segmentation], dim=1)

        skip_connections = []

        # Downsampling with CoordConv + ConvBlock + MaxPool
        for i in range(self.num_blocks):
            x = self.coord_convs[i](x)
            x = self.down_blocks[i](x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

            # Resize segmentation and concatenate again
            if i < self.num_blocks - 1:
                segmentation = F.interpolate(segmentation, size=x.shape[2:], mode='nearest')
                x = torch.cat([x, segmentation], dim=1)

        # Bottleneck processing
        x = self.bottleneck(x)

        # Upsampling with skip connections
        for i, up in enumerate(self.up_blocks):
            x = F.interpolate(x, size=skip_connections[-(i + 1)].shape[2:], mode='nearest')
            x = x + skip_connections[-(i + 1)]
            x = up(x)

        # Final output
        x = self.output_block(x)

        return x


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class AdaptiveWingLoss(torch.nn.Module):
    """Adaptive Wing Loss for robust landmark detection."""
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: Predicted heatmaps (B, C, H, W)
            target: Target heatmaps (B, C, H, W)
            weight: Optional weight map (B, 1, H, W)
        """
        delta = (target - pred).abs()

        A = self.omega * (
            1 / (1 + torch.pow(self.theta / self.epsilon,
                              self.alpha - target))
        ) * (self.alpha - target) * torch.pow(
            self.theta / self.epsilon, self.alpha - target - 1
        ) * (1 / self.epsilon)

        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - target)
        )

        losses = torch.where(
            delta < self.theta,
            self.omega * torch.log(
                1 + torch.pow(delta / self.epsilon, self.alpha - target)
            ),
            A * delta - C,
        )

        if weight is not None:
            losses = losses * weight

        return losses.mean()


custom_loss = AdaptiveWingLoss()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def landmark_pred_dist(prediction, target, get_landmarks=False):
    """Calculate mean distance between predicted and target landmarks."""
    batch_size = prediction.shape[0]

    # Apply sigmoid to prediction
    prediction = torch.sigmoid(prediction)

    # Transform to numpy arrays
    prediction = prediction.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    coord_prediction = np.zeros((batch_size, N_LANDMARKS, 2))
    coord_target = np.zeros((batch_size, N_LANDMARKS, 2))

    # Find heatmap peaks
    for j in range(batch_size):
        for i in range(N_LANDMARKS):
            try:
                coord_prediction[j, i] = ski.feature.peak_local_max(
                    prediction[j, i],
                    min_distance=100,
                    num_peaks=1,
                    exclude_border=False
                )[0] / np.array([IMAGE_HEIGHT, IMAGE_WIDTH])

                coord_target[j, i] = ski.feature.peak_local_max(
                    target[j, i],
                    min_distance=100,
                    num_peaks=1,
                    exclude_border=False
                )[0] / np.array([IMAGE_HEIGHT, IMAGE_WIDTH])

            except Exception:
                coord_prediction[j, i] = [0, 0]
                coord_target[j, i] = [0, 0]

    # Calculate mean distance
    distances = np.linalg.norm(coord_prediction - coord_target, axis=2)
    mean_distance = np.mean(distances, axis=1)

    if get_landmarks:
        return mean_distance, coord_prediction, coord_target
    else:
        return mean_distance


def apply_albumentations_batch_simple(batch_tensor: torch.Tensor, transform):
    """Apply albumentations transform to a batch of images."""
    device = batch_tensor.device
    dtype = batch_tensor.dtype

    arr = batch_tensor.detach().cpu().numpy()
    B, C, H, W = arr.shape
    assert C == 1, "This wrapper expects single-channel inputs (C=1)"

    out_list = []
    for i in range(B):
        img_hw = arr[i, 0]
        augmented = transform(image=img_hw)
        aug_img = augmented["image"]

        if isinstance(aug_img, torch.Tensor):
            aug_img = aug_img.detach().cpu().numpy()

        if aug_img.ndim == 3:
            if aug_img.shape[2] == 1:
                aug_img = np.squeeze(aug_img, axis=2)
            else:
                if aug_img.shape[0] == 1:
                    aug_img = np.squeeze(aug_img, axis=0)
                else:
                    aug_img = aug_img[..., 0]

        out_list.append(aug_img.astype(arr.dtype))

    stacked = np.stack(out_list, axis=0)
    stacked = np.expand_dims(stacked, axis=1)
    return torch.from_numpy(stacked).to(device=device, dtype=dtype)


# ============================================================================
# LOAD SEGMENTATION MODEL
# ============================================================================

print("\nLoading segmentation model...")

# Create segmentation model
segmentation_model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
).to(DEVICE)

# Load pretrained weights if specified
if PRETRAINED_MODEL_PATH and os.path.exists(PRETRAINED_MODEL_PATH):
    print(f"Loading pretrained weights from: {PRETRAINED_MODEL_PATH}")
    segmentation_model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
else:
    print("No pretrained weights specified or file not found, using ImageNet initialization")

segmentation_model.eval()

# Preview segmentation
print("Generating segmentation preview...")
with torch.no_grad():
    for batch, (img, y, weight, path) in enumerate(train_dataloader):
        img = img.to(DEVICE)
        y = y.to(DEVICE)
        weight = weight.to(DEVICE)

        # Segmentation prediction
        segment = segmentation_model(img)
        segment = torch.sigmoid(segment)

        # Apply albumentations
        segment_aug = apply_albumentations_batch_simple(segment, segment_trans)

        # Resize
        image_imgsize = TF.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
        segment_imgsize = TF.resize(segment_aug, (IMAGE_HEIGHT, IMAGE_WIDTH))
        break

if SHOW_PREVIEW or SAVE_PREVIEW:
    segmentation = segment_imgsize.cpu().detach().numpy()
    image = image_imgsize.cpu().detach().numpy()

    fig, ax = plt.subplots(1, 4, figsize=(15, 5))

    for i in range(4):
        ax[i].imshow(image[i][0], cmap="Greys_r")
        ax[i].imshow(segmentation[i][0], cmap="Reds", alpha=0.5)
        ax[i].axis("off")

    plt.tight_layout()
    
    if SAVE_PREVIEW:
        preview_path = os.path.join(OUTPUT_DIR, f"landmark_preview_fold{FOLD}.png")
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        print(f"Landmark preview saved to: {preview_path}")

    if SHOW_PREVIEW:
        plt.show()
    else:
        plt.close()

# ============================================================================
# CREATE LANDMARK MODEL
# ============================================================================

print("\nCreating landmark model...")

model = Hourglass(
    in_channels=1,
    num_blocks=HG_BLOCKS,
    intermediate_channels=64,
    output_channels=N_LANDMARKS
).to(DEVICE)

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# ============================================================================
# TRAINING SETUP
# ============================================================================

def lr_lambda(current_epoch):
    """Learning rate schedule with warmup and cosine decay."""
    if current_epoch < WARMUP_EPOCHS:
        # Linear warm-up
        return WARMUP_FACTOR + (1 - WARMUP_FACTOR) * (current_epoch / WARMUP_EPOCHS)
    else:
        # Cosine decay
        progress = (current_epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * progress))


# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Create scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_loop(dataloader, model, epoch):
    """Training loop for one epoch."""
    model.train()
    total_loss = 0
    total_IoU = 0
    total_batches = len(dataloader)

    for batch, (img, y, weight, path) in enumerate(dataloader):
        # Move to device
        img = img.to(DEVICE)
        y = y.to(DEVICE)
        weight = weight.to(DEVICE)

        # Segmentation prediction
        segment = segmentation_model(img)
        segment = torch.sigmoid(segment)

        # Apply albumentations
        segment_aug = apply_albumentations_batch_simple(segment, segment_trans)

        # Resize
        image_imgsize = TF.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
        segment_imgsize = TF.resize(segment_aug, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Landmark prediction
        pred = model(image_imgsize, segment_imgsize)
        loss = custom_loss(pred, y, weight)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Calculate metrics
        pred_sigmoid = torch.sigmoid(pred)

        total_loss += loss.item()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_sigmoid > 0.5, y > 0.5, mode='binary')
        total_IoU += smp.metrics.iou_score(tp, fp, fn, tn).mean()

    # Average metrics
    avg_loss = total_loss / total_batches
    avg_iou = total_IoU / total_batches

    print(f"Training Epoch {epoch}: Loss: {avg_loss:.3f}, IoU: {avg_iou:.3f}")

    scheduler.step()


def test_loop(dataloader, model, epoch):
    """Validation loop for one epoch."""
    model.eval()
    total_loss = 0
    total_IoU = 0
    total_dist = 0
    total_batches = len(dataloader)

    with torch.no_grad():
        for batch, (img, y, weight, path) in enumerate(dataloader):
            # Move to device
            img = img.to(DEVICE)
            y = y.to(DEVICE)
            weight = weight.to(DEVICE)

            # Segmentation prediction
            segment = segmentation_model(img)
            segment = torch.sigmoid(segment)

            # Apply albumentations
            segment_aug = apply_albumentations_batch_simple(segment, segment_trans)

            # Resize
            image_imgsize = TF.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
            segment_imgsize = TF.resize(segment_aug, (IMAGE_HEIGHT, IMAGE_WIDTH))

            # Landmark prediction
            pred = model(image_imgsize, segment_imgsize)
            loss = custom_loss(pred, y, weight)

            # Calculate metrics
            pred_sigmoid = torch.sigmoid(pred)

            total_loss += loss.item()
            tp, fp, fn, tn = smp.metrics.get_stats(pred_sigmoid > 0.5, y > 0.5, mode='binary')
            total_IoU += smp.metrics.iou_score(tp, fp, fn, tn).mean()
            total_dist += landmark_pred_dist(pred, y).mean()

    # Average metrics
    avg_loss = total_loss / total_batches
    avg_iou = total_IoU / total_batches
    avg_dist = total_dist / total_batches

    print(f"Testing {epoch}: Loss: {avg_loss:.3f}, IoU: {avg_iou:.3f}", f"Pixel Dist: {avg_dist:.3f}")
    print("-------------------")


# ============================================================================
# TRAINING
# ============================================================================

print(f"\n{'='*60}")
print("STARTING TRAINING")
print(f"{'='*60}\n")

for t in range(EPOCHS):
    train_loop(train_dataloader, model, t)
    test_loop(test_dataloader, model, t)

# ============================================================================
# SAVE MODEL
# ============================================================================

print(f"\n{'='*60}")
print("SAVING MODEL")
print(f"{'='*60}\n")

# Save full model
model_path = os.path.join(OUTPUT_DIR, f"landmark_fold-{FOLD}_{EXPERIMENT}.pth")
torch.save(model, model_path)
print(f"Full model saved to: {model_path}")

# Save model weights
weights_path = os.path.join(OUTPUT_DIR, f"landmark_weights_fold-{FOLD}_{EXPERIMENT}.pth")
torch.save(model.state_dict(), weights_path)
print(f"Model weights saved to: {weights_path}")

# Save test file list
test_files_path = os.path.join(OUTPUT_DIR, f"test_files_fold-{FOLD}_{EXPERIMENT}.npy")
np.save(test_files_path, np.asarray([str(x) for x in test_path_list]))
print(f"Test file list saved to: {test_files_path}")

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}\n")
