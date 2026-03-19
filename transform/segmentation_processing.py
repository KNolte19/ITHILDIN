"""
Segmentation processing for wing vein analysis.

This module provides functions for:
- Skeletonization of wing segmentation masks
- Junction point extraction from skeletons
- Morphological operations for skeleton refinement

COORDINATE CONVENTION:
All coordinate arrays use (2, N) format where:
- Row 0 = X coordinates (horizontal axis, column index)
- Row 1 = Y coordinates (vertical axis, row index)

This matches the convention used throughout the ITHILDIN pipeline.

Uses PlantCV library for robust skeleton processing.
"""

import numpy as np
import skimage as ski
from plantcv import plantcv as pcv


def skeletonize(segmentation: np.ndarray) -> np.ndarray:
    """
    Generate a pruned skeleton from a binary segmentation mask.

    Args:
        segmentation (np.ndarray): Input binary or probabilistic segmentation mask (H x W).

    Returns:
        np.ndarray: Pruned skeleton image (H x W), binary.
    """
    if segmentation.ndim != 2:
        raise ValueError("Expected 2D segmentation mask.")

    # Binarize segmentation
    segmentation_bin = segmentation > 0.5

    # Clean up the segmentation: remove noise
    cleaned = ski.morphology.remove_small_holes(segmentation_bin)
    cleaned = ski.morphology.remove_small_objects(cleaned)

    # Apply skeletonization
    skeleton = pcv.morphology.skeletonize(np.squeeze(cleaned))

    # Prune small branches from the skeleton
    pruned_skeleton = pcv.morphology.prune(skel_img=skeleton, size=100)[0]

    return pruned_skeleton


def extract_junction_coordinates(skeleton: np.ndarray) -> np.ndarray:
    """
    Extracts normalized junction coordinates from a skeleton.

    Args:
        skeleton (np.ndarray): Binary skeleton image (H x W).

    Returns:
        np.ndarray: Shape (2, N) with normalized junction coordinates.
                    Row 0 = X coordinates (horizontal axis, normalized to [0, 1]).
                    Row 1 = Y coordinates (vertical axis, normalized to [0, 1]).
    """
    if skeleton.ndim != 2:
        raise ValueError("Expected 2D skeleton image.")

    height, width = skeleton.shape

    # Generate junction heatmap from skeleton
    junction_heatmap = pcv.morphology.find_branch_pts(skel_img=skeleton)

    # Find junction coordinates using local maxima detection
    # Returns array of shape (N, 2) with columns [row, col] = [y, x]
    junction_coords = ski.feature.peak_local_max(
        junction_heatmap,
        min_distance=5,
        exclude_border=False
    )

    # Normalize coordinates: row/height for y, col/width for x
    # junction_coords / [height, width] gives [y_norm, x_norm]
    normalized_coords = junction_coords / np.array([height, width])
    
    # Swap columns to get [x_norm, y_norm], then transpose to get (2, N)
    # with row 0 = X, row 1 = Y
    return normalized_coords[:, [1, 0]].T
