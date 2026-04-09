"""
Landmark processing utilities for ITHILDIN wing analysis.

This module provides functions for:
- Converting heatmaps to coordinate predictions
- Aligning landmarks with skeleton junctions
- Creating semilandmarks along wing veins
- Coordinate transformations and rescaling
- Orientation adjustments

COORDINATE CONVENTION:
All coordinate arrays use (2, N) format where:
- Row 0 = X coordinates (horizontal axis, column index)
- Row 1 = Y coordinates (vertical axis, row index)

This convention is used consistently throughout the module for:
- landmark_coords, junctions_coord, consensus_coord
- All coordinate return values and parameters

Note: When interfacing with image arrays or skeleton functions that use
(row, col) indexing, coordinates are converted to (y, x) format as needed.
"""

import os

import numpy as np
import pandas as pd
import skimage as ski
from scipy.optimize import linear_sum_assignment

from config import CONFIG
from transform.image_processing import (
    process_image_with_landmarks,
    resize,
    robust_load_image,
)
from transform.wing_processing import find_skeleton_path

# ---------------------- SEGMENT TO LANDMARKS ---------------------- #

def heatmap_to_coordinates(heatmap: np.ndarray) -> np.ndarray:
    """
    Converts heatmaps to landmark coordinates by extracting the argmax position
    for each landmark and normalizing it to the range [0, 1].

    Args:
        heatmap (np.ndarray): Shape (N_landmarks, H, W), per-landmark heatmaps.

    Returns:
        np.ndarray: Shape (2, N_landmarks) with normalized coordinates.
                    Row 0 = X coordinates (horizontal axis, normalized to [0, 1]).
                    Row 1 = Y coordinates (vertical axis, normalized to [0, 1]).
    """
    N_landmarks, height, width = heatmap.shape
    coord_prediction = np.zeros((N_landmarks, 2), dtype=np.float32)

    for i in range(N_landmarks):
        try:
            # Get (row, col) of max activation
            max_index = np.unravel_index(np.argmax(heatmap[i]), heatmap[i].shape)
            # max_index is (row, col) = (y, x)
            # Normalize: row/height for y, col/width for x
            coord_prediction[i] = np.array(max_index) / [height, width]  # [y_norm, x_norm]
        except Exception:
            coord_prediction[i] = [0.0, 0.0]  # fallback on failure

    # coord_prediction is (N, 2) with columns [y, x]
    # We need to return (2, N) with row 0 = X, row 1 = Y
    # So swap columns: [:, [1, 0]] makes it [x, y], then .T makes it (2, N)
    return coord_prediction[:, [1, 0]].T


def snap_to_closest_skeleton(point: np.ndarray, skeleton: np.ndarray) -> tuple[float, float]:
    """
    Snap a point to its nearest neighbor on a skeletonized binary image.

    Args:
        point (np.ndarray): Normalized coordinates [x, y] of the point.
        skeleton (np.ndarray): 2D binary image of the skeleton.

    Returns:
        tuple[float, float]: Snapped normalized coordinates (x, y).
    """
    candidates = np.argwhere(skeleton > 0)

    if len(candidates) == 0:
        return tuple(point)

    # candidates from argwhere are (row, col) = (y, x)
    # Normalize: row/height for y, col/width for x
    candidates = candidates / [skeleton.shape[0], skeleton.shape[1]]
    # Now candidates are (N, 2) with columns [y_norm, x_norm]
    # Swap to [x_norm, y_norm] to match our point format
    candidates = candidates[:, [1, 0]]

    # Weighted Euclidean distance: emphasize x-axis
    diff = point - candidates
    diff[:, 0] *= 2  # scale x-axis difference
    distance = np.linalg.norm(diff, axis=1)

    # Return closest point as (x, y) tuple
    return tuple(candidates[np.argmin(distance)])


def consensus_coordinates(
    landmarks_coord: np.ndarray,
    junctions_coord: np.ndarray,
    skeleton: np.ndarray
) -> np.ndarray:
    """
    Aligns predicted landmarks with closest junctions using the Hungarian algorithm.
    If no junction is close enough, falls back to nearest skeleton point.

    Args:
        landmarks_coord (np.ndarray): Shape (2, N) with normalized landmarks.
                                      Row 0 = X coordinates, Row 1 = Y coordinates.
        junctions_coord (np.ndarray): Shape (2, M) with normalized junctions.
                                      Row 0 = X coordinates, Row 1 = Y coordinates.
        skeleton (np.ndarray): Binary image mask representing the skeleton.

    Returns:
        np.ndarray: Shape (2, N) with consensus coordinates.
                    Row 0 = X coordinates, Row 1 = Y coordinates.
    """
    threshold = 0.05
    large_number = 1e9

    # Compute weighted distances (2x weight for x-axis)
    # landmarks_coord[0] = X, landmarks_coord[1] = Y
    # junctions_coord[0] = X, junctions_coord[1] = Y
    dx = (landmarks_coord[0][:, None] - junctions_coord[0]) * 2
    dy = (landmarks_coord[1][:, None] - junctions_coord[1]) * 1
    distance_matrix = np.sqrt(dx ** 2 + dy ** 2)

    # Apply threshold mask for assignment
    cost_matrix = distance_matrix.copy()
    cost_matrix[cost_matrix > threshold] = large_number

    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Mark accepted assignments
    assignments = np.full(landmarks_coord.shape[1], -1, dtype=int)
    for landmark_idx, junction_idx in zip(row_ind, col_ind):
        if cost_matrix[landmark_idx, junction_idx] < large_number:
            assignments[landmark_idx] = junction_idx

    consensus_x, consensus_y = [], []

    for i, assignment in enumerate(assignments):
        if assignment == -1:
            # Snap to skeleton if no valid junction match
            # landmarks_coord[:, i] gives [x, y] for landmark i
            x, y = snap_to_closest_skeleton(landmarks_coord[:, i], skeleton)
        else:
            # Use matched junction coordinate
            x = junctions_coord[0, assignment]  # X from row 0
            y = junctions_coord[1, assignment]  # Y from row 1
        consensus_x.append(x)
        consensus_y.append(y)

    # Return as (2, N) with row 0 = X, row 1 = Y
    return np.array([consensus_x, consensus_y])

# ---------------------- SEMILANDMARKS ---------------------- #

def extract_semi_landmarks_along_path(path, num_points):
    """
    Uniformly samples semi-landmarks along a given path (excluding endpoints).

    Args:
        path (list of tuples): List of pixel coordinates in (row, col) = (y, x) format.
        num_points (int): Number of semi-landmarks to extract.

    Returns:
        list: List of [x, y] coordinate pairs for semi-landmarks in pixel space.
    """
    
    if num_points == 0 or len(path) < 2:
        return []

    indices = np.linspace(0, len(path) - 1, num=num_points + 2)[1:-1]  # skip endpoints
    # path[i] is (y, x), so path[i][1] is x and path[i][0] is y
    # Return as [x, y] pairs
    semi_lms = [[int(path[int(idx)][1]), int(path[int(idx)][0])] for idx in indices]
    return semi_lms


def create_semi_landmarks(consensus_coord, skeleton, num_landmarks_ref = CONFIG["semilandmarks_per_connection"]):
    """
    Generates semi-landmarks between landmark pairs defined in config,
    using skeleton paths and uniform sampling.

    Args:
        consensus_coord (np.ndarray): Shape (2, N) with normalized landmark coordinates.
                                      Row 0 = X coordinates (horizontal axis).
                                      Row 1 = Y coordinates (vertical axis).
        skeleton (np.ndarray): Binary skeleton image.
        num_landmarks_ref (list): Number of semilandmarks per connection.

    Returns:
        np.ndarray: Shape (2, M) with normalized coordinates of all semi-landmarks.
                    Row 0 = X coordinates, Row 1 = Y coordinates.
    """
    shortest_paths = []

    for i, connection in enumerate(CONFIG["allowed_connections"]):
        # consensus_coord[0] = X (horizontal), consensus_coord[1] = Y (vertical)
        # Convert to pixel coordinates
        x_pixels = consensus_coord[0] * CONFIG["segmentation_image_size"][0]
        y_pixels = consensus_coord[1] * CONFIG["segmentation_image_size"][1]
        
        # Create coordinate set in (row, col) = (y, x) format for array indexing
        coordinate_set = set(zip(
            y_pixels.astype(int),
            x_pixels.astype(int)
        ))

        # Convert landmark endpoints to (row, col) = (y, x) format for find_skeleton_path
        pt1 = (
            int(consensus_coord[1][connection[0]] * CONFIG["segmentation_image_size"][1]),  # y / row
            int(consensus_coord[0][connection[0]] * CONFIG["segmentation_image_size"][0]),  # x / col
        )
        pt2 = (
            int(consensus_coord[1][connection[1]] * CONFIG["segmentation_image_size"][1]),  # y / row
            int(consensus_coord[0][connection[1]] * CONFIG["segmentation_image_size"][0]),  # x / col
        )

        _, path = find_skeleton_path(
            skeleton,
            pt1,
            pt2,
            coordinate_set,
            return_shortest_path=True
        )

        num_landmarks = num_landmarks_ref[i]
        semi_lms = extract_semi_landmarks_along_path(path, num_landmarks)
        shortest_paths.extend(semi_lms)

    # semi_lms are returned as [x, y] pairs from extract_semi_landmarks_along_path
    # Normalize to [0, 1] and transpose to get shape (2, M) with row 0 = X, row 1 = Y
    return (np.array(shortest_paths) / [CONFIG["segmentation_image_size"][0], CONFIG["segmentation_image_size"][1]]).T

# ---------------------- RESCALE LANDMARKS ---------------------- #

def rescale_coordinates(consensus_coord, mask, mask_aligned, target_size=CONFIG["segmentation_image_size"]):
    """
    Maps normalized consensus coordinates back to their original spatial scale,
    accounting for image resizing, padding, and aspect ratio corrections.

    Args:
        consensus_coord (np.ndarray): Shape (2, N) with normalized consensus coordinates.
                                      Row 0 = X coordinates (horizontal axis).
                                      Row 1 = Y coordinates (vertical axis).
        mask (np.ndarray): Original binary mask.
        mask_aligned (np.ndarray): Padded/cropped version of the mask.
        target_size (tuple): Target image resolution in (W, H) format (width, height).
                            Default is CONFIG["segmentation_image_size"] = (640, 320).

    Returns:
        np.ndarray: Shape (2, N) rescaled coordinates in pixel space.
                    Row 0 = X coordinates, Row 1 = Y coordinates.
    """
    mask_aligned_shape = mask_aligned.shape
    mask_resized_shape = resize(mask_aligned, target_size[0]).shape

    ratio = mask_aligned_shape[0] / mask_resized_shape[0]  # vertical scaling

    # Apply scaling
    # consensus_coord[0] = X (horizontal), consensus_coord[1] = Y (vertical)
    # target_size[0] = width (X dimension), target_size[1] = height (Y dimension)
    scaled_x = consensus_coord[0] * target_size[0] * ratio
    scaled_y = consensus_coord[1] * target_size[1]

    # Correct for cropping and padding
    offset = target_size[0] * 0.25
    correction = (target_size[0] - mask_resized_shape[0]) * 0.5

    corrected_y = (scaled_y + offset - correction) * ratio

    # Return as (2, N) array with row 0 = X, row 1 = Y
    return np.asarray([scaled_x, corrected_y]).astype(int)

def adjust_orientation(df: pd.DataFrame):

    # Identify X-coordinate columns (covers both "X_" and "X_sm_" variants)
    x_cols = [c for c in df.columns if str(c).startswith("X_")]

    # Ensure coordinate columns are numeric; coerce any bad values to NaN
    df[x_cols] = df[x_cols].apply(pd.to_numeric, errors="coerce")

    # Build mask: invert for any orientation that is not exactly 'left'
    mask = df["Orientation"].ne("left")
        
    # Compute per-row max across X columns, ignoring NaNs
    row_max = df.loc[mask, x_cols].max(axis=1, skipna=True)

    # Right-subtract to broadcast row-wise: new_x = row_max - x
    df.loc[mask, x_cols] = df.loc[mask, x_cols].rsub(row_max, axis=0)

    return df

# ---------------------- CREATE LANDMARK DATA ---------------------- #
def create_heatmap_from_coords(landmark,
                    image_size=CONFIG["landmark_image_size"],
                    N_landmarks=CONFIG["N_landmarks"]):

    X = landmark[:,0]
    Y = landmark[:,1]
    stacked_heatmap = np.zeros((image_size[1], image_size[0], N_landmarks))
        
    for i in range(N_landmarks):
        for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    x, y = int(X[i]*image_size[0]), int(Y[i]*image_size[1])
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < stacked_heatmap.shape[0] and 0 <= nx < stacked_heatmap.shape[1]:
                        stacked_heatmap[ny, nx, i] = 1

    return stacked_heatmap


def create_landmark_heatmap(file_path, X, Y, radius=3, flipped=False, background_padding=0):
    if flipped:
        image_raw = np.fliplr(robust_load_image(file_path))
    else:
        image_raw = robust_load_image(file_path)
        
    heatmap = np.full(image_raw.shape[:2], 0, dtype=int)

    for N, (x, y) in enumerate(zip(X, Y)):
        if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < heatmap.shape[0] and 0 <= nx < heatmap.shape[1]:
                        heatmap[ny, nx] = int(N + 1)

    combined = np.concatenate([image_raw, heatmap[..., np.newaxis]], axis=-1)
    combined = (combined).astype(np.uint8)

    combined_processed = process_image_with_landmarks(combined, background_padding=background_padding)
    image_processed, heatmap_processed = combined_processed[:,:,:3], combined_processed[:,:,3]

    landmark_ls = []
    for N in range(len(X)):
        coords = np.where(heatmap_processed == N+1)
        x_coord, y_coord = coords[1][0]/CONFIG["segmentation_image_size"][0], coords[0][0]/CONFIG["segmentation_image_size"][1]
        landmark_ls.append([x_coord, y_coord])

    landmark_arr = np.asarray(landmark_ls)
    landmark_heatmap = create_heatmap_from_coords(landmark_arr, N_landmarks=len(X))

    save_path_coords = os.path.join("training", "data", "landmark", "landmark_heatmaps", file_path.split(os.sep)[-1].split(".")[0] + "_coords.npy")
    save_path_heatmap = os.path.join("training", "data", "landmark", "landmark_heatmaps", file_path.split(os.sep)[-1].split(".")[0] + "_map.npy")

    np.save(save_path_coords, landmark_arr)
    np.save(save_path_heatmap, landmark_heatmap)
    return landmark_arr, landmark_heatmap