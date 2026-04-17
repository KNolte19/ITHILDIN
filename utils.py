"""
Utility functions for ITHILDIN wing analysis.

This module provides helper functions for:
- Converting JSON prediction files to pandas DataFrames
- Visualizing images with landmark overlays
- Generating semilandmark slider configurations for geometric morphometrics
"""

import json
import os

import numpy as np
import pandas as pd
from config_loader import get_config


def json_to_dataframe(
    target_path,
    semilandmark=False,
    coordinate_type="scaled",
    family="mosquito",
    N=None,
    N_semilandmarks=None,
    with_lm_predictions=False,
):
    """
    Convert JSON prediction files to a pandas DataFrame.

    Reads all JSON files from a directory and consolidates landmark data
    into a structured DataFrame suitable for downstream analysis.

    Args:
        target_path (str): Path to directory containing JSON prediction files
        semilandmark (bool): If True, include semilandmark coordinates in output.
                            Default is False.
        coordinate_type (str): Type of coordinates to extract. Options:
                              - "scaled": Coordinates scaled to original image size
                              - "unscaled": Normalized coordinates (0-1 range)
                              - "predicted": Raw CNN predictions
                              Default is "scaled".
        N (int): Number of landmarks. Default from CONFIG.
        N_semilandmarks (int): Number of semilandmarks. Default from CONFIG.

    Returns:
        pd.DataFrame: DataFrame with columns for file metadata, predictions,
                     and landmark/semilandmark coordinates (X_i, Y_i format)

    Raises:
        ValueError: If coordinate_type is not one of the valid options
    """
    # Define default values for N and N_semilandmarks if not provided
    CONFIG = get_config(family)
    if N is None:
        N = CONFIG["N_landmarks"]
    if N_semilandmarks is None:
        N_semilandmarks = CONFIG["N_semilandmarks"]

    # Collect JSON file paths
    files = os.listdir(target_path)
    json_file_paths = [
        os.path.join(target_path, file)
        for file in files
        if file.endswith(".json") and "._" not in file
    ]

    # Determine coordinate column names based on type
    if coordinate_type == "scaled":
        coord_column = "scaled_landmark_coords"
        semi_coord_column = "scaled_semilandmark_coords"
    elif coordinate_type == "unscaled":
        coord_column = "landmark_coords"
        semi_coord_column = "semilandmark_coords"
    elif coordinate_type == "predicted":
        coord_column = "predicted_coords"
        semilandmark = False  # No semilandmarks for predicted coords
    else:
        raise ValueError(
            "Invalid coordinate_type. Choose from 'scaled', 'unscaled', or 'predicted'."
        )

    # Initialize DataFrame for landmark data
    base_columns = [
        "File",
        "Status",
        "Centroid",
        "Orientation",
        "CNN_Predicted_Taxa",
        "CNN_Predicted_Score",
    ]

    coord_columns = [f"X_{i}" for i in range(N)] + [f"Y_{i}" for i in range(N)]
    landmark_columns = ["LM_Predicted_Taxa", "LM_Predicted_Score", "ENS_Predicted_Taxa", "ENS_Predicted_Score"]

    if with_lm_predictions == True:
        coord_columns += landmark_columns
    
    df = pd.DataFrame(columns=base_columns + coord_columns)

    # Process each JSON file
    for i, path in enumerate(json_file_paths):
        try:
            with open(path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            row = [
                json_data["file_name"],
                json_data["status"],
                json_data["scaled_centroid"],
                json_data["orientation"],
                json_data["cnn_prediction"]["top"],
                json_data["cnn_prediction"]["score"],
            ] + list(np.array(json_data[coord_column]).flatten())
            
            if with_lm_predictions == True:
                row += [
                    json_data["landmark_prediction"]["top"],
                    json_data["landmark_prediction"]["score"],
                    json_data["ensemble_prediction"]["top"],
                    json_data["ensemble_prediction"]["score"],
                ]
            
            df.loc[i] = row

        except Exception as e:
            print(f"Error processing {path}: {e}")
            print(df.columns)
            # Create row with NaN values for failed processing
            # Use safe access in case json_data wasn't loaded
            file_name = json_data.get("file_name", path) if "json_data" in locals() else path
            status = json_data.get("status", "Error") if "json_data" in locals() else "Error"
            df.loc[i] = [file_name, status] + [np.nan] * (len(df.columns) - 2)

    landmark_df = df.copy()

    # Add semilandmark data if requested
    if semilandmark:
        # Initialize DataFrame for semilandmark data
        semi_coord_columns = [f"X_sm_{i}" for i in np.arange(N, N + N_semilandmarks)] + [
            f"Y_sm_{i}" for i in np.arange(N, N + N_semilandmarks)
        ]

        if with_lm_predictions == True:
            semi_coord_columns += landmark_columns

        semilandmark_df = pd.DataFrame(columns=base_columns + semi_coord_columns)

        for i, path in enumerate(json_file_paths):
            with open(path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # Only include semilandmarks if status is valid and data is complete
            if json_data["status"] not in ["Failed Repair", "Repair Error"]:
                semi_coords = list(np.array(json_data[semi_coord_column]).flatten())
                if len(semi_coords) == int(N_semilandmarks * 2):
                    row = [
                        json_data["file_name"],
                        json_data["status"],
                        json_data["scaled_centroid"],
                        json_data["orientation"],
                        json_data["cnn_prediction"]["top"],
                        json_data["cnn_prediction"]["score"],
                    ] + semi_coords

                    if with_lm_predictions == True:
                        row += [
                                json_data["landmark_prediction"]["top"],
                                json_data["landmark_prediction"]["score"],
                                json_data["ensemble_prediction"]["top"],
                                json_data["ensemble_prediction"]["score"],
                            ]

                    semilandmark_df.loc[i] = row

                else:
                    semilandmark_df.loc[i] = [
                        json_data["file_name"],
                        json_data["status"],
                    ] + [np.nan] * (len(semilandmark_df.columns) - 2)
            else:
                semilandmark_df.loc[i] = [json_data["file_name"], json_data["status"]] + [
                    np.nan
                ] * (len(semilandmark_df.columns) - 2)

        # Merge semilandmark coordinates into main DataFrame
        semilandmark_coords = semilandmark_df[
            [col for col in semilandmark_df.columns if col.startswith(("X_sm_", "Y_sm_"))]
        ]
        for col in semilandmark_coords.columns:
            landmark_df[col] = semilandmark_coords[col]

    # Format and clean up the DataFrame
    landmark_df["File"] = landmark_df["File"].astype(str)
    landmark_df["File_Name"] = [file.split("/")[-1] for file in landmark_df["File"]]

    # Convert coordinate columns to numeric and round appropriately
    for col in landmark_df.columns:
        if col.startswith(("X_", "Y_")) and coordinate_type == "scaled":
            landmark_df[col] = pd.to_numeric(landmark_df[col], errors="coerce")
        if col == "CNN_Predicted_Score":
            landmark_df[col] = pd.to_numeric(landmark_df[col], errors="coerce").round(2)
        if col == "LDA_Predicted_Score":
            landmark_df[col] = pd.to_numeric(landmark_df[col], errors="coerce").round(2)
        if col == "ENS_Predicted_Score":
            landmark_df[col] = pd.to_numeric(landmark_df[col], errors="coerce").round(2)
        if col == "Centroid":
            landmark_df[col] = (
                pd.to_numeric(landmark_df[col], errors="coerce")
                .round(0)
                .astype("Int64")
            )

    return landmark_df


def save_image_with_landmarks(image, save_path, landmarks, semilandmarks):
    """
    Save an image with landmarks and semilandmarks visualized as overlays.

    Creates a visualization showing the input image with landmarks (red dots)
    and semilandmarks (cyan dots) overlaid at their predicted positions.
    Uses PIL for rendering to avoid any matplotlib backend / display dependency.

    Args:
        image (np.ndarray): Input image array, float32 in [0, 1] range,
                            shape (H, W) for grayscale or (H, W, C) for color.
        save_path (str): Output file path for saving the visualization.
        landmarks (np.ndarray): Shape (2, N) with landmark coordinates.
                                Row 0 = X coordinates (pixel space).
                                Row 1 = Y coordinates (pixel space).
        semilandmarks (np.ndarray): Shape (2, M) with semilandmark coordinates.
                                    Row 0 = X coordinates (pixel space).
                                    Row 1 = Y coordinates (pixel space).

    Returns:
        None. Saves the figures to the specified paths.
    """
    from PIL import Image as PILImage, ImageDraw

    def _prepare_image_uint8_rgb(image_array):
        image_array = np.asarray(image_array)

        if image_array.ndim == 2:
            image_array = image_array[:, :, np.newaxis]
        elif image_array.ndim != 3:
            image_array = np.squeeze(image_array)
            if image_array.ndim == 2:
                image_array = image_array[:, :, np.newaxis]
            elif image_array.ndim != 3:
                raise ValueError("Unsupported image shape for save_image_with_landmarks")

        if image_array.shape[2] == 1:
            image_array = np.repeat(image_array, 3, axis=2)
        elif image_array.shape[2] >= 3:
            image_array = image_array[:, :, :3]
        else:
            raise ValueError("Unsupported channel count for save_image_with_landmarks")

        is_float = np.issubdtype(image_array.dtype, np.floating)
        image_array = np.nan_to_num(image_array, nan=0.0, posinf=255.0, neginf=0.0)
        if is_float and image_array.size > 0 and image_array.max() <= 1.0:
            image_array = image_array * 255.0

        return np.clip(image_array, 0, 255).astype(np.uint8)

    def _prepare_xy_points(coords):
        coords = np.asarray(coords)
        if coords.size == 0:
            return np.empty((0, 2), dtype=np.float64)

        coords = np.squeeze(coords)
        if coords.ndim == 1:
            if coords.shape[0] != 2:
                return np.empty((0, 2), dtype=np.float64)
            coords = coords.reshape(1, 2)
        elif coords.ndim != 2:
            return np.empty((0, 2), dtype=np.float64)

        if coords.shape[1] == 2:
            return coords.astype(np.float64, copy=False)
        if coords.shape[0] == 2:
            return coords.T.astype(np.float64, copy=False)
        return np.empty((0, 2), dtype=np.float64)

    def _draw_point(draw_ctx, x, y, radius, color, width, height):
        if not np.isfinite(x) or not np.isfinite(y):
            return

        cx = int(round(float(x)))
        cy = int(round(float(y)))
        if cx < 0 or cx >= width or cy < 0 or cy >= height:
            return

        radius = max(1, int(radius))
        x0 = max(cx - radius, 0)
        y0 = max(cy - radius, 0)
        x1 = min(cx + radius, width - 1)
        y1 = min(cy + radius, height - 1)

        if x1 < x0 or y1 < y0:
            return

        draw_ctx.ellipse([x0, y0, x1, y1], fill=color)

    img_uint8 = _prepare_image_uint8_rgb(image)
    pil_img = PILImage.fromarray(img_uint8, mode="RGB")

    # Save raw image (no landmarks)
    raw_path = save_path.replace(".png", "_raw.png")
    pil_img.save(raw_path)

    # Draw landmarks and semilandmarks onto a copy
    overlay = pil_img.copy()
    draw = ImageDraw.Draw(overlay)

    h, w = img_uint8.shape[:2]
    min_dim = max(1, min(h, w))
    lm_radius = max(1, int(round(min_dim * 0.006)))
    slm_radius = max(1, int(round(min_dim * 0.004)))

    lm_points = _prepare_xy_points(landmarks)
    slm_points = _prepare_xy_points(semilandmarks)

    # Semilandmarks first so landmarks render on top
    for x, y in slm_points:
        _draw_point(draw, x, y, slm_radius, (0, 255, 255), w, h)

    for x, y in lm_points:
        _draw_point(draw, x, y, lm_radius + 1, (255, 255, 255), w, h)
        _draw_point(draw, x, y, lm_radius, (255, 0, 0), w, h)

    overlay.save(save_path)


def generate_sliders(family="mosquito"):
    """
    Generate semilandmark slider configurations for geometric morphometrics.

    Creates triplet definitions for sliding semilandmarks along curves
    between fixed landmarks. Each triplet defines [before, sliding, after]
    points for use in Procrustes analysis with semilandmarks.

    Returns:
        pd.DataFrame: DataFrame where each row is a triplet [point_before,
                     sliding_point, point_after] defining how a semilandmark
                     should slide along its curve.

    Note:
        Uses configuration from CONFIG including:
        - allowed_connections: List of (start, end) landmark pairs
        - semilandmarks_per_connection: Number of semilandmarks per connection
        - N_landmarks: Number of fixed landmarks
    """
    CONFIG = get_config(family)
    allowed_connections = CONFIG["allowed_connections"]
    semilandmarks_per_connection = CONFIG["semilandmarks_per_connection"]
    N_fixed = CONFIG["N_landmarks"]

    current_semi_idx = N_fixed + 1
    sliders = []

    for connection, count in zip(allowed_connections, semilandmarks_per_connection):
        # Get R-style indices for start and end landmarks (1-indexed)
        start_node = connection[0] + 1
        end_node = connection[1] + 1

        # Generate semilandmark indices for this curve
        path_semis = list(range(current_semi_idx, current_semi_idx + count))
        full_path = [start_node] + path_semis + [end_node]

        # Update index counter for next connection
        current_semi_idx += count

        # Generate triplets for each sliding point
        # Format: [point_before, sliding_point, point_after]
        for i in range(1, len(full_path) - 1):
            triplet = [full_path[i - 1], full_path[i], full_path[i + 1]]
            sliders.append(triplet)

    # Convert to DataFrame for R compatibility
    df_sliders = pd.DataFrame(sliders)
    return df_sliders
