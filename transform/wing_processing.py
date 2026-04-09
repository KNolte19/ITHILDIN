"""
Wing skeleton processing and repair utilities.

This module provides functions for:
- Validating wing skeleton completeness
- Repairing disconnected skeleton segments
- Path finding along wing veins
- Skeleton connectivity checking

COORDINATE CONVENTION:
For internal numpy arrays representing coordinates, we use (2, N) format:
- Row 0 = X coordinates (horizontal axis, column index)
- Row 1 = Y coordinates (vertical axis, row index)

However, when working with image arrays and path-finding functions,
coordinates are represented as tuples in (row, col) = (y, x) format
for proper array indexing.

Uses breadth-first search and morphological operations to ensure
that landmark positions are properly connected via wing vein skeletons.
"""

from collections import deque
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from plantcv import plantcv as pcv

from config import CONFIG

# Define 8-connected neighbor offsets for morphological operations
NEIGHBOR_OFFSETS_8 = [(-1, -1), (-1, 0), (-1, 1),
                      ( 0, -1),          ( 0, 1),
                      ( 1, -1), ( 1, 0), ( 1, 1)]


def get_neighbors(pt_list):
    """
    Return all unique 8-connected neighbors for a list of points.

    Args:
        pt_list (iterable): List of (y, x) coordinates.

    Returns:
        tuple: 8-connected neighbor coordinates.
    """
    neighbors = []
    for point in pt_list:
        for dr, dc in NEIGHBOR_OFFSETS_8:
            neighbors.append((point[0] + dr, point[1] + dc))
    return tuple(neighbors)

def scale_coord(x, y):
    """
    Scale normalized coordinates to image pixel space.
    
    This function converts normalized coordinates (in range [0, 1]) to pixel coordinates.
    
    Args:
        x (float): Normalized x-coordinate (horizontal axis, range [0, 1]).
        y (float): Normalized y-coordinate (vertical axis, range [0, 1]).

    Returns:
        tuple: (row, col) in pixel space, i.e., (y_pixel, x_pixel).
               Note: Returns (row, col) format for array indexing where row = y, col = x.
    """
    # Convert normalized (x, y) to pixel (col, row) = (x_pixel, y_pixel)
    # Return as (row, col) for array indexing: (y_pixel, x_pixel)
    x_pixel = int(x * CONFIG["segmentation_image_size"][0])
    y_pixel = int(y * CONFIG["segmentation_image_size"][1])
    return y_pixel, x_pixel

def find_skeleton_path(skel, pt1, pt2, coordinate_set, return_shortest_path=False):
    """
    Search for a valid path between pt1 and pt2 in the skeleton using BFS.

    Args:
        skel (np.ndarray): Binary skeleton image.
        pt1, pt2 (tuple): Start and end points (y, x).
        coordinate_set (set): Other junction coordinates to avoid.
        return_shortest_path (bool): If True, return path list as well.

    Returns:
        bool or (bool, list): Path found flag or (flag, path list).
    """
    skel = skel.astype(bool)
    coordinate_set = set(coordinate_set)
    coordinate_set.discard(pt1)
    coordinate_set.discard(pt2)

    rows, cols = skel.shape

    # Avoid 2-level neighborhood of all other junction points
    forbidden = set(get_neighbors(get_neighbors(coordinate_set)))

    # Define local bounding box
    buffer = 30
    y_max, y_min = max(pt1[0], pt2[0]) + buffer, min(pt1[0], pt2[0]) - buffer
    x_max, x_min = max(pt1[1], pt2[1]) + buffer, min(pt1[1], pt2[1]) - buffer

    visited = {pt1}
    parent = {pt1: None}
    queue = deque([pt1])

    while queue:
        current = queue.popleft()
        if current == pt2:
            if return_shortest_path:
                path = []
                while current:
                    path.append(current)
                    current = parent[current]
                return True, list(reversed(path))
            return True

        for dr, dc in NEIGHBOR_OFFSETS_8:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if not skel[nr, nc] or neighbor in visited or neighbor in forbidden:
                continue
            if nr < y_min or nr > y_max or nc < x_min or nc > x_max:
                continue
            visited.add(neighbor)
            parent[neighbor] = current
            queue.append(neighbor)

    return (False, []) if return_shortest_path else False

def repair_skeleton(skeleton, pt1, pt2, segmentation_logit, coordinate_set, skeleton_paths, buffer=30):
    """
    Repair a missing path between pt1 and pt2 using the segmentation logit as a cost map.

    Args:
        skeleton (np.ndarray): Binary skeleton image.
        pt1, pt2 (tuple): Start and end coordinates in (y, x).
        segmentation_logit (np.ndarray): Probabilistic segmentation output.
        coordinate_set (set): Junction points to avoid.
        skeleton_paths (dict): Previously validated paths.
        buffer (int): Spatial corridor width for search.

    Returns:
        (np.ndarray, RepairStatus): Updated skeleton and status.
    """
    epsilon = 0.1
    scaled = (segmentation_logit - segmentation_logit.min()) / (segmentation_logit.max() - segmentation_logit.min())
    cost_map = 1.0 / (scaled + epsilon)

    coordinate_set.discard(pt1)
    coordinate_set.discard(pt2)

    # Mask around junctions
    forbidden = np.array(list(get_neighbors(get_neighbors(coordinate_set))))
    cost_map[forbidden[:, 0], forbidden[:, 1]] = cost_map.max() #TODO

    # Exclude all other skeleton paths
    skeleton_paths = {k: v for k, v in skeleton_paths.items() if k != (pt1, pt2)}
    all_path_coords = set((x, y) for group in skeleton_paths.values() for (x, y) in group)
    forbidden = np.array(list(get_neighbors(get_neighbors(all_path_coords))))
    cost_map[forbidden[:, 0], forbidden[:, 1]] = cost_map.max() #TODO

    # Limit to a search corridor
    ymin, ymax = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])
    xmin, xmax = min(pt1[1], pt2[1]), max(pt1[1], pt2[1])

    # Clamp Values to Image Bounds
    ymin_mask = max(0, ymin - buffer)
    ymax_mask = min(cost_map.shape[0], ymax + buffer)
    xmin_mask = max(0, xmin - buffer)
    xmax_mask = min(cost_map.shape[1], xmax + buffer)

    allowed_mask = np.zeros_like(cost_map, dtype=bool)
    allowed_mask[ymin_mask:ymax_mask, xmin_mask:xmax_mask] = 1
    cost_map[~allowed_mask] = cost_map.max()

    try:
        path, _ = ski.graph.route_through_array(
            cost_map, start=pt1, end=pt2, fully_connected=True, geometric=True
        )
    except Exception as e:
        return skeleton, "Error"

    # Draw new path
    repaired = np.zeros_like(skeleton)
    for y, x in path:
        repaired[y, x] = 255

    repaired_skeleton = ski.morphology.skeletonize(skeleton + repaired) > 0
    status = "Repaired"

    return (repaired_skeleton * 255).astype(np.uint8), status

def check_skeleton(skeleton, consensus_coord, segmentation_logit):
    """
    Check skeleton for missing connections and repair as needed.

    Args:
        skeleton (np.ndarray): Input skeleton image.
        consensus_coord (np.ndarray): Shape (2, N) with normalized coordinates.
                                      Row 0 = X coordinates (horizontal axis).
                                      Row 1 = Y coordinates (vertical axis).
        segmentation_logit (np.ndarray): Segmentation output for cost guidance.

    Returns:
        (np.ndarray, RepairStatus): Final skeleton and status outcome.
    """
    skeleton_check = {}
    skeleton_paths = {}
    status = "Full Skeleton"

    # Precompute junctions as pixel coordinates
    # consensus_coord[0] = X (horizontal), consensus_coord[1] = Y (vertical)
    coord_x = consensus_coord[0] * CONFIG["segmentation_image_size"][0]
    coord_y = consensus_coord[1] * CONFIG["segmentation_image_size"][1]
    # For array indexing, we need (row, col) = (y, x) format
    original_coordinate_set = set(zip(coord_y.astype(int), coord_x.astype(int)))

    # Check all required connections
    for connection in CONFIG["allowed_connections"]:
        pt1 = scale_coord(consensus_coord[0][connection[0]], consensus_coord[1][connection[0]])
        pt2 = scale_coord(consensus_coord[0][connection[1]], consensus_coord[1][connection[1]])
        coordinate_set = original_coordinate_set.copy()

        check, path = find_skeleton_path(skeleton, pt1, pt2, coordinate_set, return_shortest_path=True)
        skeleton_check[(pt1, pt2)] = check
        skeleton_paths[(pt1, pt2)] = path

    for (pt1, pt2), connected in skeleton_check.items():
        if connected:
            continue  # No repair needed

        # Attempt to repair missing connection
        coordinate_set = original_coordinate_set.copy()
        skeleton, status = repair_skeleton(skeleton, pt1, pt2, segmentation_logit, coordinate_set, skeleton_paths)

        if status == "Error":
            return skeleton, "Failed Repair", (connection[0], connection[1])

        status = "Repaired"
        coordinate_set = original_coordinate_set.copy()
        check, path = find_skeleton_path(skeleton, pt1, pt2, coordinate_set, return_shortest_path=True)
        skeleton_check[(pt1, pt2)] = check
        skeleton_paths[(pt1, pt2)] = path

        # Ensure no undesired connections were introduced
        for connection in CONFIG["not_allowed_connections"]:
            pt1 = scale_coord(consensus_coord[0][connection[0]], consensus_coord[1][connection[0]])
            pt2 = scale_coord(consensus_coord[0][connection[1]], consensus_coord[1][connection[1]])
            coordinate_set = original_coordinate_set.copy()

            check, _ = find_skeleton_path(skeleton, pt1, pt2, coordinate_set, return_shortest_path=True)
            if check:
                status = "Failed Repair"
                return skeleton, status, (connection[0], connection[1])

    return skeleton, status, None
