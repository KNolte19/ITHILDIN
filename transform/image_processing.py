"""
Image processing utilities for ITHILDIN wing analysis.

This module provides functions for:
- Loading and preprocessing wing images
- Background removal using segmentation models
- Image alignment and normalization
- Contrast enhancement (CLAHE)
- Geometric transformations (resize, pad, crop)

All functions are designed to work with numpy arrays and scikit-image.
"""

import imghdr

import numpy as np
import skimage.exposure
import skimage.filters
import skimage.io as ski_io
import skimage.measure
import skimage.morphology
import skimage.transform
import tifffile
import torch
from PIL import Image, UnidentifiedImageError
from rembg import remove

from config import CONFIG

def robust_load_image(file_path, force_resave_tiff=True):
    """
    Loads an image with robust TIFF support.

    If the file isn't a readable TIFF, attempts to convert from common formats (PNG, JPEG, etc.)
    and optionally resaves as TIFF for compatibility.

    Ensures the returned array is HxWx3 (repeats grayscale channel or strips alpha).

    Args:
        file_path (str): Path to the image file.
        force_resave_tiff (bool): Whether to overwrite and resave non-TIFFs as TIFF.

    Returns:
        np.ndarray: The loaded image as a NumPy array with 3 channels.

    Raises:
        ValueError: If the format is unsupported or the file is corrupted.
    """
    try:
        img = ski_io.imread(file_path)
    except (tifffile.TiffFileError, KeyError):
        real_format = imghdr.what(file_path)

        if real_format in {"png", "jpeg", "bmp", "gif"}:
            try:
                with Image.open(file_path) as pil:
                    pil = pil.convert("RGB")
                    if force_resave_tiff:
                        pil.save(file_path, format="TIFF")
                        print(f"Resaved {file_path} as proper TIFF.")
                        img = ski_io.imread(file_path)
                    else:
                        img = np.array(pil)
            except UnidentifiedImageError:
                raise ValueError(f"Cannot identify image format: {file_path}")
        else:
            raise ValueError(f"Unsupported or corrupted image: {file_path}")

    # Ensure image has 3 channels: repeat grayscale or drop alpha if present
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def resize(image, target_width):
    """
    Resize image to target width while maintaining aspect ratio.

    Args:
        image (np.ndarray): Input image array
        target_width (int): Desired output width in pixels

    Returns:
        np.ndarray: Resized image with same number of channels as input
    """
    ratio = image.shape[0] / image.shape[1]
    target_height = int(target_width * ratio)

    # Use different interpolation for boolean masks vs regular images
    if image.any().dtype == bool or image.dtype == np.bool_:
        image_resized = skimage.transform.resize(
            image,
            (target_height, target_width),
            anti_aliasing=False,
            preserve_range=True,
            order=0,  # Nearest-neighbor for binary
        )
    else:
        image_resized = skimage.transform.resize(
            image,
            (target_height, target_width),
            anti_aliasing=False,
            order=1,  # Bilinear for grayscale/color
        )

    return image_resized
    

def pad_to_square(image):
    """
    Pads the image with zeros to make it square.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Padded square image.
    """
    height, width = image.shape[:2]
    max_dim = max(height, width)

    pad_height = (max_dim - height) // 2
    pad_width = (max_dim - width) // 2

    pad_vals = (
        (pad_height, max_dim - height - pad_height),
        (pad_width, max_dim - width - pad_width),
        (0, 0) if image.ndim == 3 else (0, 0)
    )

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    image_padded = np.pad(image, pad_vals, mode='constant', constant_values=0)

    return image_padded


def remove_background(image, return_mask=True, bg_session=None):
    """
    Remove background using AI-based segmentation (rembg).

    Extracts the foreground object by using alpha channel masking and
    keeping only the largest connected component.

    Args:
        image (np.ndarray): Input RGB image (H x W x 3 or H x W x 4)
        return_mask (bool): If True, return both image and mask. Default True.
        bg_session: Optional rembg session for faster repeated calls

    Returns:
        tuple or np.ndarray:
            If return_mask=True: (foreground_image, binary_mask)
            If return_mask=False: foreground_image only
    """
    rgba_image = remove(image[:, :, :3], bgcolor=(0, 0, 0, -1), session=bg_session)

    alpha_channel = np.asarray(rgba_image)[:, :, -1]
    # Threshold alpha to get initial mask
    initial_mask = alpha_channel > 10

    # Label connected regions
    labeled_mask = skimage.measure.label(initial_mask)
    regions = skimage.measure.regionprops(labeled_mask)

    if not regions:
        mask = np.zeros_like(alpha_channel, dtype=bool)
    else:
        # Find the largest region (assumed to be the wing)
        largest_region = max(regions, key=lambda r: r.area)
        mask = labeled_mask == largest_region.label

    masked_rgba_image = np.asarray(rgba_image) * np.dstack([mask] * 4)
    rgb_image = masked_rgba_image[:, :, :-1]  # Remove alpha channel

    # Preserve landmark channel if present in original image
    if image.shape[2] == 4:
        landmark_channel = image[:, :, 3]
        rgb_image = np.concatenate([rgb_image, landmark_channel[..., np.newaxis]], axis=-1)

    if return_mask:
        return rgb_image, mask

    return rgb_image


def align(image, mask, background_padding=0):
    """
    Align image so major axis of foreground object is vertical.

    Rotates the image to make the wing's major axis point upward,
    then crops to the bounding box of the rotated mask.

    Args:
        image (np.ndarray): RGB or grayscale image
        mask (np.ndarray): Binary mask of foreground region
        background_padding (int): Extra pixels to include around bounding box.
                                 Default is 0.

    Returns:
        tuple: (aligned_image, aligned_mask) both cropped to bounding box
    """
    labeled = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(labeled)

    if not regions:
        return image, mask

    # Find largest region and calculate rotation angle
    largest_region = max(regions, key=lambda r: r.axis_major_length)
    angle = -(largest_region.orientation * 180 / np.pi + 90)
    if angle < -90:
        angle += 180

    # Rotate image and mask
    rotated_img = skimage.transform.rotate(
        image, angle, resize=False, mode="constant", preserve_range=True, order=0
    )
    rotated_mask = (
        skimage.transform.rotate(
            mask.astype(float), angle, resize=False, mode="constant", preserve_range=True, order=0
        )
        > 0
    )

    # Find bounding box of rotated mask
    coords = np.argwhere(rotated_mask)
    rmin, cmin = coords.min(axis=0)
    rmax, cmax = coords.max(axis=0) + 1

    # Apply optional padding around bounding box
    if background_padding and background_padding > 0:
        pad = int(background_padding)
        rmin_pad = max(rmin - pad, 0)
        cmin_pad = max(cmin - pad, 0)
        rmax_pad = min(rmax + pad, rotated_mask.shape[0])
        cmax_pad = min(cmax + pad, rotated_mask.shape[1])
    else:
        rmin_pad, cmin_pad, rmax_pad, cmax_pad = rmin, cmin, rmax, cmax

    image_cropped = rotated_img[rmin_pad:rmax_pad, cmin_pad:cmax_pad]
    mask_cropped = rotated_mask[rmin_pad:rmax_pad, cmin_pad:cmax_pad]

    return image_cropped, mask_cropped

def crop_to_ratio(image, height_ratio=0.5, width_ratio=1.0):
    """
    Crops the image to a specified height:width ratio from center.

    Args:
        image (np.ndarray): Input image.
        height_ratio (float): Ratio of height to keep.
        width_ratio (float): Ratio of width to keep.

    Returns:
        np.ndarray: Cropped image.
    """
    h, w = image.shape[:2]

    # Calculate the new image size
    new_h = int(h * height_ratio)
    new_w = int(w * width_ratio)

    # Calculate the center of the image
    center_h = h // 2
    center_w = w // 2

    # Calculate the cropping box
    crop_h = new_h // 2
    crop_w = new_w // 2

    # Crop the image
    image_cropped = image[center_h - crop_h:center_h + crop_h, center_w - crop_w:center_w + crop_w]

    return image_cropped


def CLAHE(image, clip_limit=0.1, nbins=128, strong=False):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Enhances local contrast while preventing over-amplification of noise.

    Args:
        image (np.ndarray): RGB or grayscale image
        clip_limit (float): Clipping limit for contrast enhancement. Default 0.1.
        nbins (int): Number of histogram bins. Default 128.
        strong (bool): If True, apply stronger enhancement (clip=0.6, nbins=48).
                      Default False.

    Returns:
        np.ndarray: Contrast-enhanced grayscale image, with median filtering applied
    """
    # Convert to grayscale if needed
    gray_image = np.mean(image, axis=-1) if image.ndim == 3 else image

    if strong:
        image = skimage.exposure.equalize_adapthist(gray_image, clip_limit=0.6, nbins=48)
    else:
        image = skimage.exposure.equalize_adapthist(
            gray_image, clip_limit=clip_limit, nbins=nbins
        )

    # Apply median filter to reduce noise
    image = skimage.filters.median(image, skimage.morphology.disk(2))

    return image


def process_image(file, from_stream=False, background_padding=0):
    """
    Complete image preprocessing pipeline.

    Performs background removal, alignment, and normalization.

    Args:
        file: Either file path (str) or file stream object
        from_stream (bool): If True, treat file as stream. Default False.
        background_padding (int): Pixels to pad around bounding box. Default 0.

    Returns:
        tuple: (aligned_image, aligned_mask, original_mask)
            - aligned_image: Preprocessed image normalized to [0, 1]
            - aligned_mask: Binary mask after alignment
            - original_mask: Binary mask before alignment
    """
    if from_stream:
        file = file.stream
        image_raw = np.array(Image.open(file, mode="r").convert("RGB"))
    else:
        image_raw = robust_load_image(file)

    image_rembg, mask = remove_background(image_raw)
    image_aligned, mask_aligned = align(
        image_rembg, mask, background_padding=background_padding
    )
    image_aligned = (image_aligned / 255.0).astype(np.float32)

    return image_aligned, mask_aligned, mask


def transform_image(
    image,
    mask_aligned,
    contrast="Soft",
    resize_size=CONFIG["segmentation_image_size"][0],
):
    """
    Apply final transformations to prepare image for model input.

    Sequence: mask application -> resize -> pad to square -> crop to ratio -> CLAHE

    Args:
        image (np.ndarray): Input image (normalized to [0, 1])
        mask_aligned (np.ndarray): Binary mask to apply
        contrast (str or None): Contrast enhancement mode. Options:
                               - None: No enhancement (returns RGB image)
                               - "Soft": Standard CLAHE (default)
                               - "Strong": Aggressive CLAHE
        resize_size (int): Target width for resizing. Default from CONFIG.

    Returns:
        np.ndarray: Transformed image ready for model input.
                   If contrast is None: RGB image
                   If contrast is set: Grayscale image with CLAHE applied

    Raises:
        ValueError: If contrast is not None, "Soft", or "Strong"
    """
    # Apply mask
    image = image * np.dstack([mask_aligned] * 3)

    # Geometric transformations
    image = resize(image, resize_size)
    image = pad_to_square(image)
    image = crop_to_ratio(image)

    # Regenerate mask after transformations
    mask_realigned = image[:, :, 0] != 0

    # Apply contrast enhancement if requested
    if contrast is None:
        return image
    elif contrast == "Soft":
        clahe_image = CLAHE(image) * mask_realigned
        return clahe_image
    elif contrast == "Strong":
        clahe_image = CLAHE(image, strong=True) * mask_realigned
        return clahe_image
    else:
        raise ValueError(
            "Invalid contrast option. Must be None, 'Soft', or 'Strong'."
        )


def process_image_with_landmarks(image_raw, background_padding=0):
    """
    Process image preserving landmark annotations (if present).

    Similar to process_image but designed for training/validation
    where landmark channels may be present.

    Args:
        image_raw (np.ndarray): Raw input image (may have 4th channel for landmarks)
        background_padding (int): Padding around bounding box. Default 0.

    Returns:
        np.ndarray: Processed image with geometric transformations applied
    """
    image_rembg, mask = remove_background(image_raw)
    image_aligned, mask_aligned = align(
        image_rembg, mask, background_padding=background_padding
    )

    image = resize(image_aligned, CONFIG["segmentation_image_size"][0])

    image = pad_to_square(image)
    image = crop_to_ratio(image)

    return image
