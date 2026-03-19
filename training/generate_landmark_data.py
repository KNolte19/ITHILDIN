#!/usr/bin/env python3
"""
Generate Landmark Training Data

This script processes raw wing images and their corresponding landmark annotations
to create training data for the landmark detection model. It performs image alignment,
landmark heatmap generation, and creates numpy arrays ready for model training.

The script outputs four .npy files:
    - forlandmark_{experiment}_paths.npy: Array of image filenames
    - forlandmark_{experiment}_images.npy: Array of normalized image data
    - forlandmark_{experiment}_heatmap.npy: Array of landmark heatmaps

Author: ITHILDIN Wing Analysis Project
License: CC BY-NC 4.0
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import skimage as ski
from tqdm import tqdm

try:
    # Try importing config from the script's directory or parent
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(script_dir.parent))
    from config import CONFIG
    from transform import image_processing, landmark_processing
except ImportError as e:
    print(
        "\nERROR: Could not import CONFIG from config.py.\n"
        "Make sure config.py exists in the project root or the training/ folder.\n"
        "Original error: ", e
    )
    sys.exit(1)




def process_images(experiment_name, root_path, image_size, annotations_df, background_padding=20, verbose=True):
    """
    Process raw images by applying alignment, normalization.
    
    This function loads raw images, applies the standard image processing pipeline.
    
    Args:
        experiment_name (str): Name of the experiment/dataset (e.g., 'data_droso', 'data_tsetse')
        root_path (str): Root path of the project
        image_size (int): Target width for the processed images
        annotations_df (pd.DataFrame): DataFrame with filename and coordinate annotations
        background_padding (int): Padding around the detected wing in pixels
        verbose (bool): Whether to print progress information
        
    Returns:
        int: Number of images successfully processed
    """
    image_folder = os.path.join(root_path, "training", experiment_name, "landmark", "image_raw")
    processed_folder = os.path.join(root_path, "training", experiment_name, "landmark", "image_processed")
    
    # Check if input folder exists
    if not os.path.exists(image_folder):
        raise FileNotFoundError(
            f"Input folder not found: {image_folder}\n"
            f"Please ensure raw images are placed in: training/{experiment_name}/landmark/image_raw/"
        )
    
    # Create output folder if it doesn't exist
    os.makedirs(processed_folder, exist_ok=True)
    
    # List all files in image folder
    files = os.listdir(image_folder)
    
    # Only select image files
    valid_extensions = [".png", ".tiff", ".tif", ".jpg", ".jpeg"]
    files_cleaned = [file for file in files if any(file.lower().endswith(ext) for ext in valid_extensions)]
    
    if len(files_cleaned) == 0:
        raise ValueError(
            f"No image files found in {image_folder}\n"
            f"Supported formats: {', '.join(valid_extensions)}"
        )
    
    if verbose:
        print(f"Found {len(files_cleaned)} images to process")
        print(f"Input folder: {image_folder}")
        print(f"Output folder: {processed_folder}")
        print(f"Background padding: {background_padding} pixels")
    
    # Provide lists with input and output paths
    image_paths = [os.path.join(root_path, image_folder, file) for file in files_cleaned]
    processed_paths = [os.path.join(root_path, processed_folder, Path(file).stem + ".png") for file in files_cleaned]
    
    # Process all images
    successful = 0
    skipped = 0
    iterator = tqdm(range(len(image_paths)), desc="Processing images") if verbose else range(len(image_paths))
    
    for i in iterator:
        try:
            # Skip if already processed
            if os.path.exists(processed_paths[i]):
                skipped += 1
                successful += 1
                continue
            
            # Apply image processing pipeline with padding
            image_aligned, mask_aligned, mask = image_processing.process_image(
                image_paths[i], 
                background_padding=background_padding
            )
            image = image_processing.transform_image(
                image_aligned, 
                mask_aligned, 
                contrast=None, 
                resize_size=image_size
            )
            
            
            # Save the processed image
            ski.io.imsave(processed_paths[i], (image * 255).astype(np.uint8))
            successful += 1
            
        except Exception as e:
            print(f"ERROR processing {files_cleaned[i]}: {e}")
    
    if verbose:
        print(f"\nProcessing complete:")
        print(f"  Successfully processed: {successful - skipped}/{len(image_paths)}")
        print(f"  Already processed (skipped): {skipped}")
        print(f"  Total ready: {successful}/{len(image_paths)}")
    
    return successful

def process_landmarks(experiment_name, root_path, annotations_df, n_landmarks, background_padding=20, verbose=True):
    """
    Create landmark heatmaps from coordinate annotations.
    
    This function reads landmark coordinates from the annotations CSV and generates
    heatmap representations for each image.
    
    Args:
        experiment_name (str): Name of the experiment/dataset
        root_path (str): Root path of the project
        annotations_df (pd.DataFrame): DataFrame with landmark coordinates
        n_landmarks (int): Number of landmarks to process
        background_padding (int): Padding around the detected wing in pixels
        verbose (bool): Whether to print progress information
        
    Returns:
        int: Number of heatmaps successfully created
    """
    heatmap_folder = os.path.join(root_path, "training", experiment_name, "landmark", "landmark_heatmaps")
    
    # Create output folder if it doesn't exist
    os.makedirs(heatmap_folder, exist_ok=True)
    
    df = annotations_df.reset_index(drop=True)
    
    if verbose:
        print(f"\nCreating landmark heatmaps...")
        print(f"Number of annotations: {len(df)}")
        print(f"Number of landmarks per image: {n_landmarks}")
        print(f"Output folder: {heatmap_folder}")
    
    successful = 0
    skipped = 0
    iterator = tqdm(range(len(df)), desc="Creating heatmaps") if verbose else range(len(df))
    
    for i in iterator:
        df_row = df.iloc[i]
        
        # Construct image path
        filename = df_row["filename"].replace(" ", "")
        image_folder = os.path.join(root_path, "training", experiment_name, "landmark", "image_raw")
        
        # Try to find the image with correct extension
        image_path = None
        for ext in [".jpg", ".png", ".tiff", ".tif", ".jpeg", ".JPG", ".PNG"]:
            test_path = os.path.join(image_folder, Path(filename).stem + ext)
            if os.path.exists(test_path):
                image_path = test_path
                break
        
        if image_path is None:
            if verbose and successful < 5:
                print(f"  WARNING: Image not found for {filename}")
            continue
        
        save_path_heatmap = os.path.join(
            heatmap_folder, 
            Path(filename).stem + "_map.npy"
        )
        
        # Skip if already processed
        if os.path.exists(save_path_heatmap):
            skipped += 1
            successful += 1
            continue
        
        try:
            # Extract landmark coordinates
            x_cols = [col for col in df.columns if col.startswith("X_")]
            y_cols = [col for col in df.columns if col.startswith("Y_")]
            
            X = np.asarray(df_row[x_cols].astype(int))
            Y = np.asarray(df_row[y_cols].astype(int))

            # Create heatmap with padding
            landmark_arr, landmark_heatmap = landmark_processing.create_landmark_heatmap(
                        image_path, X, Y, flipped=False, background_padding=background_padding
                    )
                
            # Save heatmap and coordinates
            np.save(save_path_heatmap, landmark_heatmap)
            np.save(save_path_heatmap.replace("_map.npy", "_coords.npy"), landmark_arr)
            successful += 1
            
        except Exception as e:
               print(f"ERROR creating heatmap for {filename}: {e}")
    
    if verbose:
        print(f"\nHeatmap creation complete:")
        print(f"  Successfully created: {successful - skipped}/{len(df)}")
        print(f"  Already created (skipped): {skipped}")
        print(f"  Total ready: {successful}/{len(df)}")
    
    return successful


def get_common_files(experiment_name, root_path, annotations_df, verbose=True):
    """
    Find images that have all required data: processed image and heatmap.
    
    Args:
        experiment_name (str): Name of the experiment/dataset
        root_path (str): Root path of the project
        annotations_df (pd.DataFrame): DataFrame with filenames
        verbose (bool): Whether to print progress information
        
    Returns:
        list: Sorted list of common base filenames
    """
    processed_folder = os.path.join(root_path, "training", experiment_name, "landmark", "image_processed")
    heatmap_folder = os.path.join(root_path, "training", experiment_name, "landmark", "landmark_heatmaps")
    
    # List files in each folder
    images = [file for file in os.listdir(processed_folder) if file.endswith(".png")]
    heatmaps = [file for file in os.listdir(heatmap_folder) if file.endswith("_map.npy")]
    
    # Get basenames (without extension)
    reference_basenames = set([Path(file).stem.replace(" ", "") for file in annotations_df["filename"].tolist()])
    image_basenames = set([Path(file).stem for file in images])
    heatmap_basenames = set([file.replace("_map.npy", "") for file in heatmaps])
    
    # Find intersection
    common_basenames = sorted(image_basenames & heatmap_basenames & reference_basenames)
    
    if verbose:
        print(f"\nFinding common files:")
        print(f"  Processed images: {len(image_basenames)}")
        print(f"  Heatmaps: {len(heatmap_basenames)}")
        print(f"  Annotations: {len(reference_basenames)}")
        print(f"  Common files: {len(common_basenames)}")
    
    return common_basenames


def create_training_arrays(experiment_name, root_path, common_files, n_landmarks, 
                          image_height=240, image_width=480, verbose=True):
    """
    Create numpy arrays for training from processed data.
    
    Args:
        experiment_name (str): Name of the experiment/dataset
        root_path (str): Root path of the project
        common_files (list): List of common base filenames
        n_landmarks (int): Number of landmarks
        image_height (int): Target image height for heatmaps
        image_width (int): Target image width for heatmaps
        verbose (bool): Whether to print progress information
        
    Returns:
        tuple: (number of samples, paths to saved arrays)
    """
    if len(common_files) == 0:
        raise ValueError("No common files found to create training arrays")
    
    # Image dimensions - processed images are stored at 640x320 (standard processing size)
    PROCESSED_IMAGE_HEIGHT = 320
    PROCESSED_IMAGE_WIDTH = 640
    
    if verbose:
        print(f"\nCreating training arrays...")
        print(f"  Number of samples: {len(common_files)}")
        print(f"  Processed image size: {PROCESSED_IMAGE_HEIGHT} x {PROCESSED_IMAGE_WIDTH}")
        print(f"  Heatmap target size: {image_height} x {image_width}")
    
    processed_folder = os.path.join(root_path, "training", experiment_name, "landmark", "image_processed")
    heatmap_folder = os.path.join(root_path, "training", experiment_name, "landmark", "landmark_heatmaps")
    
    # Initialize arrays
    image_arr = np.zeros((len(common_files), PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH, 3), dtype=np.uint8)
    heatmap_arr = np.zeros((len(common_files), image_height, image_width, n_landmarks), dtype=np.uint8)
    
    # Load images
    if verbose:
        print("\n  Loading images...")
    for i, base_name in enumerate(tqdm(common_files, desc="Loading images") if verbose else common_files):
        image_path = os.path.join(processed_folder, base_name + ".png")
        image = ski.io.imread(image_path)
        # Resize to standard processing size
        image_resized = (ski.transform.resize(image, (PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH), anti_aliasing=True) * 255).astype(np.uint8)
        image_arr[i] = image_resized
    
    # Load heatmaps
    if verbose:
        print("  Loading heatmaps...")
    for i, base_name in enumerate(tqdm(common_files, desc="Loading heatmaps") if verbose else common_files):
        heatmap_path = os.path.join(heatmap_folder, base_name + "_map.npy")
        heatmap = np.load(heatmap_path)
        if heatmap.shape[2] > n_landmarks:
            heatmap = heatmap[:, :, :n_landmarks]
        heatmap_arr[i] = heatmap
    
    # Save arrays
    output_dir = os.path.join(root_path, "training", experiment_name)
    path_arr = np.asarray(list(common_files))
    
    path_file = os.path.join(output_dir, f"forlandmark_{experiment_name}_paths.npy")
    image_file = os.path.join(output_dir, f"forlandmark_{experiment_name}_images.npy")
    heatmap_file = os.path.join(output_dir, f"forlandmark_{experiment_name}_heatmap.npy")
    
    np.save(path_file, path_arr)
    np.save(image_file, image_arr)
    np.save(heatmap_file, heatmap_arr)
    
    if verbose:
        print(f"\nSuccessfully created training data:")
        print(f"  Number of samples: {len(common_files)}")
        print(f"  Image shape: {image_arr.shape}")
        print(f"  Heatmap shape: {heatmap_arr.shape}")
        print(f"\nSaved files:")
        print(f"  {path_file}")
        print(f"  {image_file}")
        print(f"  {heatmap_file}")
    
    return len(common_files), (path_file, image_file, heatmap_file)


def main():
    """Main function to run the landmark data generation pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate training data for landmark detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default dataset (data_tsetse with 11 landmarks)
  python generate_landmark_data.py --experiment data_tsetse --n-landmarks 11
  
  # Process Drosophila dataset (14 landmarks, remove landmark 4)
  python generate_landmark_data.py --experiment data_droso --n-landmarks 14
  
  # Only process images (skip heatmap and array creation)
  python generate_landmark_data.py --process-only --experiment data_tsetse --n-landmarks 11
  
  # Only create arrays (images and heatmaps already processed)
  python generate_landmark_data.py --arrays-only --experiment data_tsetse --n-landmarks 11
  
  # Use custom padding
  python generate_landmark_data.py --experiment data_tsetse --n-landmarks 11 --background-padding 30

For more information, see the README.md in this folder.
        """
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of the experiment/dataset folder (e.g., data_tsetse, data_droso)"
    )
    
    parser.add_argument(
        "--n-landmarks",
        type=int,
        required=True,
        help="Number of landmarks to process (e.g., 11 for tsetse, 14 for droso)"
    )
    
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Path to annotations CSV file (default: training/{experiment}/landmark/landmark_annotations.csv)"
    )
    
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="Root path of the project (default: from CONFIG)"
    )
    
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Target image width in pixels (default: from CONFIG, typically 640)"
    )
    
    parser.add_argument(
        "--background-padding",
        type=int,
        default=20,
        help="Padding around detected wing in pixels (default: 20)"
    )
    
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Only process images and create heatmaps, don't create training arrays"
    )
    
    parser.add_argument(
        "--arrays-only",
        action="store_true",
        help="Only create training arrays, skip image processing and heatmap creation"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    root_path = args.root_path if args.root_path else CONFIG["root_path"]
    image_size = args.image_size if args.image_size else CONFIG["segmentation_image_size"][0]
    verbose = not args.quiet
    
    # Determine annotations file path
    if args.annotations:
        annotations_path = args.annotations
    else:
        annotations_path = os.path.join(
            root_path, "training", args.experiment, "landmark", "landmark_annotations.csv"
        )
    
    if verbose:
        print("=" * 70)
        print("ITHILDIN Wing Analysis - Landmark Training Data Generation")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Experiment: {args.experiment}")
        print(f"  Number of landmarks: {args.n_landmarks}")
        print(f"  Root path: {root_path}")
        print(f"  Image size: {image_size} x {image_size // 2}")
        print(f"  Background padding: {args.background_padding} pixels")
        print(f"  Annotations file: {annotations_path}")
        print(f"  Mode: {'Process only' if args.process_only else 'Arrays only' if args.arrays_only else 'Full pipeline'}")
        print()
    
    try:
        # Load annotations
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(
                f"Annotations file not found: {annotations_path}\n"
                f"Please ensure the CSV file exists with columns: filename, X_0, Y_0, ..., X_{args.n_landmarks - 1}, Y_{args.n_landmarks - 1}"
            )
        
        if verbose:
            print(f"Loading annotations from: {annotations_path}")
        
        annotations_df = pd.read_csv(annotations_path, sep=";")
        
        # Validate required columns
        required_cols = ["filename"] + [f"{axis}_{i}" for i in range(args.n_landmarks) for axis in ["X", "Y"]]
        missing_cols = [col for col in required_cols if col not in annotations_df.columns]
        if missing_cols:
            raise ValueError(
                f"Annotations file missing required columns: {missing_cols}\n"
            )
        
        if verbose:
            print(f"Loaded {len(annotations_df)} annotations")
        
        # Step 1: Process images
        if not args.arrays_only:
            if verbose:
                print("\n" + "=" * 70)
                print("STEP 1: Processing Images")
                print("=" * 70)
            
            num_processed = process_images(
                args.experiment, root_path, image_size, annotations_df, 
                args.background_padding, verbose
            )
            
            if num_processed == 0:
                print("ERROR: No images were processed successfully")
                return 1
            
            # Step 2: Create landmark heatmaps
            if verbose:
                print("\n" + "=" * 70)
                print("STEP 2: Creating Landmark Heatmaps")
                print("=" * 70)
            
            num_heatmaps = process_landmarks(
                args.experiment, root_path, annotations_df, args.n_landmarks,
                args.background_padding, verbose
            )
            
            if num_heatmaps == 0:
                print("ERROR: No heatmaps were created successfully")
                return 1
        
        # Step 3: Create training arrays
        if not args.process_only:
            if verbose:
                print("\n" + "=" * 70)
                print("STEP 3: Creating Training Arrays")
                print("=" * 70)
            
            common_files = get_common_files(args.experiment, root_path, annotations_df, verbose)
            
            if len(common_files) == 0:
                print("ERROR: No common files found with complete data")
                print("Ensure each image has:")
                print("  1. Processed image in landmark/image_processed/")
                print("  2. Heatmap in landmark/landmark_heatmaps/")
                print("  3. Entry in annotations CSV")
                return 1
            
            num_samples, output_files = create_training_arrays(
                args.experiment, root_path, common_files, args.n_landmarks,
                verbose=verbose
            )
            
            if num_samples == 0:
                print("ERROR: No training samples were created")
                return 1
        
        if verbose:
            print("\n" + "=" * 70)
            print("SUCCESS: Landmark training data generation completed!")
            print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
