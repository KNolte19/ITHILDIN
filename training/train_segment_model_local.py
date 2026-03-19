"""
ITHILDIN Segmentation Model Training Script (Local Version)

This script trains a wing segmentation model using UNet++ architecture with EfficientNet-b0 encoder.
It's designed to run locally with direct access to the training data in the repository.

For Colab version with Google Drive integration, use train_segment_model_colab.ipynb instead.

Author: ITHILDIN Project
"""

import os
import sys
import torch
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from skimage.morphology import skeletonize, dilation
from sklearn.model_selection import KFold
import torch.nn as nn

# ============================================================================
# CONFIGURATION PARAMETERS - Modify these for your experiment
# ============================================================================

# Experiment Settings
EXPERIMENT = "data_droso"  # Your experiment name
DATA_PATH = "./training/data_droso"  # Path to training data directory

# Training Hyperparameters
LEARNING_RATE = 1e-3  # Initial learning rate
EPOCHS = 2  # Number of training epochs
BATCH_SIZE = 8  # Batch size for training
IMAGE_SIZE = 640  # Input image size (square)

# Loss Function Parameters
ALPHA = 0.25  # Weight balance between general loss and skeleton loss

# Learning Rate Scheduler Parameters
WARMUP_EPOCHS = 8  # Number of warmup epochs
WARMUP_FACTOR = 0.05  # Starting learning rate multiplier for warmup

# Cross-Validation Settings
K_FOLDS = 5  # Number of folds for cross-validation
FOLD = 0  # Which fold to train on (0 to K_FOLDS-1)

# Model Settings
PRETRAINED_BOOL = False  # Whether to use pretrained weights
PRETRAINED_MODEL_PATH = None  # Path to pretrained model (if PRETRAINED_BOOL is True)

# Output Settings
OUTPUT_DIR = "./training/data_droso"  # Directory to save trained models

# Hardware Settings
DEVICE = None  # Will auto-detect if None (cuda/mps/cpu)

# Visualization Settings
SHOW_PREVIEW = False  # Set to True to display training data preview
SAVE_PREVIEW = True  # Set to True to save preview images

# ============================================================================
# SETUP
# ============================================================================

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
print(f"ITHILDIN SEGMENTATION MODEL TRAINING")
print(f"{'='*60}")
print(f"Experiment: {EXPERIMENT}")
print(f"Data Path: {DATA_PATH}")
print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Image Size: {int(IMAGE_SIZE*.5)}x{IMAGE_SIZE}")
print(f"Fold: {FOLD}/{K_FOLDS}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"{'='*60}\n")

# ============================================================================
# DATASET CLASS
# ============================================================================

class CustomDataset(Dataset):
    """
    Custom Dataset for segmentation training.
    
    Args:
        file_path_input: Path to .npy file containing input images
        file_path_output: Path to .npy file containing segmentation masks
        input_transforms: Transforms to apply to input images
        output_transforms: Transforms to apply to output masks
        augment_transforms: Geometric augmentation transforms
        augment_bool: Whether to apply augmentations
    """
    def __init__(self, file_path_input, file_path_output, input_transforms,
                 output_transforms, augment_transforms, augment_bool=False):
        self.input = np.load(file_path_input).astype(np.float32)
        self.output = np.load(file_path_output).astype(np.float32)
        self.input_transforms = input_transforms
        self.output_transforms = output_transforms
        self.augment_transforms = augment_transforms
        self.augment_bool = augment_bool

    def __len__(self):
        return len(self.input)

    def geo_transform(self, input_img, output_img):
        """Apply geometric augmentations consistently to image and mask."""
        # Random rotation
        angle = transforms.RandomRotation.get_params((-5, 5))
        input_img = TF.rotate(input_img, angle)
        output_img = TF.rotate(output_img, angle)

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            input_img = TF.hflip(input_img)
            output_img = TF.hflip(output_img)

        # Random shear
        params = transforms.RandomAffine.get_params(
            degrees=(0, 0), translate=(0, 0),
            scale_ranges=(1, 1), shears=(-15, 15),
            img_size=(int(IMAGE_SIZE * 0.5), IMAGE_SIZE)
        )
        shear_x, shear_y = params[-1]
        input_img = TF.affine(input_img, angle=params[0], translate=params[1],
                             scale=params[2], shear=(shear_x, shear_y))
        output_img = TF.affine(output_img, angle=params[0], translate=params[1],
                              scale=params[2], shear=(shear_x, shear_y))

        return input_img, output_img

    def CLAHE_transform(self, image):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        # Reduce dimension
        image = torch.mean(image, dim=0).numpy()
        
        # Apply CLAHE
        clahe_image = ski.exposure.equalize_adapthist(image, clip_limit=.1, nbins=128)

        # Use median filter to reduce noise
        clahe_image = ski.filters.median(clahe_image, ski.morphology.disk(2))

        return torch.tensor(clahe_image, dtype=torch.float32)

    def __getitem__(self, idx):
        input_img = self.input[idx]
        output_seg = self.output[idx]

        # Convert to tensors
        input_img = self.input_transforms(input_img)
        output_seg = self.output_transforms(output_seg)

        # Apply geometric augmentations if enabled
        if self.augment_bool:
            input_img, output_seg = self.geo_transform(input_img, output_seg)
            input_img = self.augment_transforms(input_img)

        # Apply CLAHE
        input_img = torch.clamp(input_img, 0, 1)
        input_img = self.CLAHE_transform(input_img).unsqueeze(0) 

        return input_img, output_seg

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading training data...")

# Construct data paths
input_path = os.path.join(DATA_PATH, f"forsegment_{EXPERIMENT}_images.npy")
output_path = os.path.join(DATA_PATH, f"forsegment_{EXPERIMENT}_segments.npy")
path_path = os.path.join(DATA_PATH, f"forsegment_{EXPERIMENT}_paths.npy")

# Check if files exist
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input data not found: {input_path}")
if not os.path.exists(output_path):
    raise FileNotFoundError(f"Output data not found: {output_path}")

# Create KFold splitter
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# Load data to get length
temp_data = np.load(path_path)
n_samples = len(temp_data)
del temp_data

print(f"Total samples: {n_samples}")

# Generate all fold splits
splits = list(kf.split(np.arange(n_samples)))

# Select the specified fold
train_idx, test_idx = splits[FOLD]

print(f"Training samples: {len(train_idx)}")
print(f"Testing samples: {len(test_idx)}")

# ============================================================================
# DATA TRANSFORMATIONS AND AUGMENTATION
# ============================================================================

input_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE*.5, IMAGE_SIZE))
])

output_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE*.5, IMAGE_SIZE))
])  

augment_transforms = transforms.Compose([
    v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    v2.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5)),
    v2.RandomPosterize(bits=8, p=0.1),
    v2.RandomAutocontrast(p=0.1),
    v2.RandomAdjustSharpness(sharpness_factor=2.0, p=0.2),
    v2.RandomGrayscale(p=0.2),
])

# ============================================================================
# CREATE DATASETS AND DATALOADERS
# ============================================================================

print("\nCreating datasets and dataloaders...")

# Create datasets
train_dataset = CustomDataset(
    file_path_input=input_path,
    file_path_output=output_path,
    input_transforms=input_transforms,
    output_transforms=output_transforms,
    augment_transforms=augment_transforms,
    augment_bool=True
)

test_dataset = CustomDataset(
    file_path_input=input_path,
    file_path_output=output_path,
    input_transforms=input_transforms,
    output_transforms=output_transforms,
    augment_transforms=None,
    augment_bool=False
)

# Create dataloaders with subset samplers for the fold
from torch.utils.data import SubsetRandomSampler

train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    sampler=test_sampler,
)

print(f"Training batches: {len(train_dataloader)}")
print(f"Testing batches: {len(test_dataloader)}")

# ============================================================================
# PREVIEW DATA (Optional)
# ============================================================================

if SHOW_PREVIEW or SAVE_PREVIEW:
    print("\nGenerating data preview...")
    X_batch, y_batch = next(iter(train_dataloader))
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(4):
        axes[0, i].imshow(X_batch[i][0], cmap='gray')
        axes[0, i].set_title(f"Input Image {i+1}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(y_batch[i][0], cmap='gray')
        axes[1, i].set_title(f"Segmentation Mask {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if SAVE_PREVIEW:
        preview_path = os.path.join(OUTPUT_DIR, f"segmentation_training_preview_fold{FOLD}.png")
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        print(f"Preview saved to: {preview_path}")
    
    if SHOW_PREVIEW:
        plt.show()
    else:
        plt.close()

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def get_model():
    """Create and return the segmentation model."""
    # Create base UNet++ model
    base_model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet" if not PRETRAINED_BOOL else None,
        in_channels=1,  # Grayscale input
        classes=1  # Binary segmentation
    )

    model = base_model

    return model


print("\nCreating segmentation model...")
model = get_model().to(DEVICE)

if PRETRAINED_BOOL and PRETRAINED_MODEL_PATH and os.path.exists(PRETRAINED_MODEL_PATH):
    print(f"Loading pretrained weights from: {PRETRAINED_MODEL_PATH}")
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def skeletonize_mask(mask, threshold=0.5):
    """Create skeleton from mask for skeleton-aware loss."""
    # Create boolean mask
    bool_mask = mask[0].cpu().detach().numpy() > threshold

    # Skeletonize
    skeleton = skeletonize(bool_mask)

    # Create tube around skeleton within original mask
    skeleton = dilation(dilation(skeleton)) * bool_mask

    # Convert to tensor
    skeleton = skeleton.astype(np.int16) > 0

    return torch.tensor(skeleton, dtype=torch.float32)


def skeleton_recall_loss(pred, target, smooth=1e-6):
    """Calculate skeleton recall loss."""
    # Skeletonize masks
    pred_skeleton = skeletonize_mask(pred)
    target_skeleton = skeletonize_mask(target)

    # Calculate soft true positives
    soft_true_positives = torch.sum((pred_skeleton * target_skeleton))

    # Calculate actual positives
    actual_positives = torch.sum(target_skeleton)

    # Compute soft recall
    soft_recall_value = (soft_true_positives + smooth) / (actual_positives + smooth)

    return 1 - soft_recall_value


def soft_dice_loss(pred, target, smooth=1e-6):
    """Calculate Dice loss."""
    # Calculate intersection
    intersection = (pred * target).sum()

    # Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice


def focal_loss(pred, target, gamma=2, alpha=0.25, smooth=1e-6):
    """Calculate focal loss based on Dice coefficient."""
    # Calculate intersection
    intersection = (pred * target).sum()

    # Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    # Calculate focal loss
    focal = -alpha * (1 - dice) ** gamma * torch.log(dice)

    return focal


def combined_loss(pred, target, alpha):
    """Combined loss function with focal and skeleton recall components."""
    # Apply sigmoid activation
    pred = torch.sigmoid(pred)

    # Calculate general loss (focal)
    general_loss = focal_loss(pred, target)

    # Calculate skeleton recall loss
    skeleton_loss = skeleton_recall_loss(pred, target)

    # Combine losses
    return alpha * general_loss + (1 - alpha) * skeleton_loss


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

# Define scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_loop(dataloader, model, epoch):
    """Training loop for one epoch."""
    model.train()
    total_loss = 0
    total_IoU = 0
    total_soft_dice_loss = 0
    total_soft_recall_loss = 0
    total_batches = len(dataloader)

    for batch, (X, y) in enumerate(dataloader):
        # Move to device
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        # Forward pass
        pred = model(X)
        loss = combined_loss(pred, y, alpha=ALPHA)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Calculate metrics
        pred = torch.sigmoid(pred)

        total_loss += loss.item()
        tp, fp, fn, tn = smp.metrics.get_stats(pred > 0.5, y > 0.5, mode='binary')
        total_IoU += smp.metrics.iou_score(tp, fp, fn, tn).mean()
        total_soft_dice_loss += soft_dice_loss(pred, y)
        total_soft_recall_loss += skeleton_recall_loss(pred, y)

    # Average metrics
    avg_loss = total_loss / total_batches
    avg_dice = total_soft_dice_loss / total_batches
    avg_recall = total_soft_recall_loss / total_batches
    avg_iou = total_IoU / total_batches

    print(f"Epoch {epoch}: Loss: {avg_loss:.3f}, Dice: {avg_dice:.3f}, "
          f"SoftSkel-Recall: {avg_recall:.3f}, IoU: {avg_iou:.3f}")

    scheduler.step()


def test_loop(dataloader, model, epoch):
    """Validation loop for one epoch."""
    model.eval()
    total_loss = 0
    total_IoU = 0
    total_soft_dice_loss = 0
    total_soft_recall_loss = 0
    total_batches = len(dataloader)

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # Move to device
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            # Forward pass
            pred = model(X)
            loss = combined_loss(pred, y, alpha=ALPHA)

            # Calculate metrics
            pred = torch.sigmoid(pred)

            total_loss += loss.item()
            tp, fp, fn, tn = smp.metrics.get_stats(pred > 0.5, y > 0.5, mode='binary')
            total_IoU += smp.metrics.iou_score(tp, fp, fn, tn).mean()
            total_soft_dice_loss += soft_dice_loss(pred, y)
            total_soft_recall_loss += skeleton_recall_loss(pred, y)

    # Average metrics
    avg_loss = total_loss / total_batches
    avg_dice = total_soft_dice_loss / total_batches
    avg_recall = total_soft_recall_loss / total_batches
    avg_iou = total_IoU / total_batches

    print(f"Validation: Loss: {avg_loss:.3f}, Dice: {avg_dice:.3f}, "
          f"SoftSkel-Recall: {avg_recall:.3f}, IoU: {avg_iou:.3f}")
    print("-----------------------------------------------------")


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
model_path = os.path.join(OUTPUT_DIR, f"segmentation_fold-{FOLD}.pth")
torch.save(model, model_path)
print(f"Full model saved to: {model_path}")

# Save model weights
weights_path = os.path.join(OUTPUT_DIR, f"segmentation_weights_fold-{FOLD}.pth")
torch.save(model.state_dict(), weights_path)
print(f"Model weights saved to: {weights_path}")

# ============================================================================
# EVALUATION
# ============================================================================

print(f"\n{'='*60}")
print("EVALUATING MODEL")
print(f"{'='*60}\n")


def evaluate_model(dataloader, model):
    """Evaluate model and return metrics."""
    model.eval()
    total_loss = 0
    total_IoU = 0
    total_soft_dice_loss = 0
    total_soft_recall_loss = 0
    total_batches = len(dataloader)

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(X)
            loss = combined_loss(pred, y, alpha=ALPHA)

            pred = torch.sigmoid(pred)

            total_loss += loss.item()
            tp, fp, fn, tn = smp.metrics.get_stats(pred > 0.5, y > 0.5, mode='binary')
            total_IoU += smp.metrics.iou_score(tp, fp, fn, tn).mean()
            total_soft_dice_loss += soft_dice_loss(pred, y)
            total_soft_recall_loss += skeleton_recall_loss(pred, y)

    avg_loss = total_loss / total_batches
    avg_dice = total_soft_dice_loss / total_batches
    avg_recall = total_soft_recall_loss / total_batches
    avg_iou = total_IoU / total_batches

    metrics = {
        "loss": avg_loss,
        "dice_loss": avg_dice.item(),
        "skeleton_recall_loss": avg_recall.item(),
        "iou": avg_iou.item()
    }

    return metrics


metrics = evaluate_model(test_dataloader, model)

# Save metrics
metrics_path = os.path.join(OUTPUT_DIR, f"evaluation_metrics_fold-{FOLD}.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print("Evaluation Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value:.4f}")
print(f"\nMetrics saved to: {metrics_path}")

# ============================================================================
# VISUALIZATION
# ============================================================================

if SHOW_PREVIEW or SAVE_PREVIEW:
    print("\nGenerating prediction visualization...")
    
    model.eval()
    test_input, ground_truth = next(iter(test_dataloader))
    test_output = model(test_input.to(DEVICE))

    fig, ax = plt.subplots(4, 4, figsize=(12, 12))
    
    for i in range(min(4, len(test_input))):
        ax[i][0].imshow(test_input[i][0], cmap='gray')
        ax[i][0].set_title("Input Image")
        ax[i][0].axis('off')

        ax[i][1].imshow(torch.sigmoid(test_output[i][0]).detach().cpu().numpy(), cmap="gist_yarg")
        ax[i][1].set_title("Predicted Mask")
        ax[i][1].axis('off')

        ax[i][2].imshow(ground_truth[i][0], cmap="gist_yarg")
        ax[i][2].set_title("Ground Truth")
        ax[i][2].axis('off')

        diff = ground_truth[i][0] - torch.sigmoid(test_output[i][0]).detach().cpu().numpy()
        ax[i][3].imshow(diff, cmap='seismic', vmin=-1, vmax=1)
        ax[i][3].set_title("Difference")
        ax[i][3].axis('off')

    plt.tight_layout()
    
    if SAVE_PREVIEW:
        viz_path = os.path.join(OUTPUT_DIR, f"predictions_fold-{FOLD}.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"Predictions visualization saved to: {viz_path}")
    
    if SHOW_PREVIEW:
        plt.show()
    else:
        plt.close()

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}\n")
