#!/usr/bin/env python3
"""
Generate Segmentation Training Data

This script processes raw wing images and their corresponding segmentation masks
to create training data for the wing segmentation model. It performs image alignment,
normalization, and creates numpy arrays ready for model training.

The script outputs three .npy files:
    - path_forsegment.npy: Array of image filenames
    - image_forsegment.npy: Array of normalized image data
    - segment_forsegment.npy: Array of normalized segmentation masks

Author: ITHILDIN Wing Analysis Project
License: CC BY-NC 4.0
"""

import argparse
import os
import sys
import numpy as np
import skimage as ski
from tqdm import tqdm
from pathlib import Path


# --- Add this block to ensure config.py is found ---
try:
    # Try importing config from the script's directory or parent
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(script_dir.parent))
    from config import CONFIG
    from transform import image_processing
except ImportError as e:
    print(
        "\nERROR: Could not import CONFIG from config.py.\n"
        "Make sure config.py exists in the project root or the training/ folder.\n"
        "Original error: ", e
    )
    sys.exit(1)


def process_images(experiment_name, root_path, image_size, verbose=True):
    """
    Process raw images by applying alignment and normalization.
    
    This function loads raw images from the image_raw folder, applies the standard
    image processing pipeline (alignment, background removal, normalization), and
    saves the processed images as PNG files.
    
    Args:
        experiment_name (str): Name of the experiment/dataset (e.g., 'data_droso', 'data_tsetse')
        root_path (str): Root path of the project
        image_size (int): Target width for the processed images (height will be half of width)
        verbose (bool): Whether to print progress information
        
    Returns:
        int: Number of images successfully processed
    """
    image_folder = os.path.join(root_path, "training", experiment_name, "segmentation", "image_raw")
    processed_folder = os.path.join(root_path, "training", experiment_name, "segmentation", "image_processed")
    
    # Check if input folder exists
    if not os.path.exists(image_folder):
        raise FileNotFoundError(
            f"Input folder not found: {image_folder}\n"
            f"Please ensure raw images are placed in: training/{experiment_name}/segmentation/image_raw/"
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
    
    # Provide lists with input and output paths
    image_paths = [os.path.join(root_path, image_folder, file) for file in files_cleaned]
    processed_paths = [os.path.join(root_path, processed_folder, Path(file).stem + ".png") for file in files_cleaned]
    
    # Process all images
    successful = 0
    iterator = tqdm(range(len(image_paths)), desc="Processing images") if verbose else range(len(image_paths))
    
    for i in iterator:
        try:
            # Skip if already processed
            if os.path.exists(processed_paths[i]):
                if verbose and i < 5:  # Only print for first few
                    print(f"  Skipping {files_cleaned[i]} (already processed)")
                successful += 1
                continue
            
            # Apply image processing pipeline
            image_aligned, mask_aligned, mask = image_processing.process_image(image_paths[i])
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
        print(f"\nSuccessfully processed {successful}/{len(image_paths)} images")
    
    return successful


def create_training_arrays(experiment_name, root_path, verbose=True):
    """
    Create numpy arrays for training from processed images and segmentation masks.
    
    This function matches processed images with their corresponding segmentation masks,
    normalizes them, and saves them as numpy arrays for model training.
    
    Args:
        experiment_name (str): Name of the experiment/dataset
        root_path (str): Root path of the project
        verbose (bool): Whether to print progress information
        
    Returns:
        tuple: (number of samples, paths to saved arrays)
    """
    processed_folder = os.path.join(root_path, "training", experiment_name, "segmentation", "image_processed")
    segment_folder = os.path.join(root_path, "training", experiment_name, "segmentation", "segmentation_fromannotator")
    
    # Check if folders exist
    if not os.path.exists(processed_folder):
        raise FileNotFoundError(
            f"Processed images folder not found: {processed_folder}\n"
            f"Please run image processing first."
        )
    
    if not os.path.exists(segment_folder):
        raise FileNotFoundError(
            f"Segmentation masks folder not found: {segment_folder}\n"
            f"Please ensure segmentation masks are placed in: training/{experiment_name}/segmentation/segmentation_fromannotator/"
        )
    
    # List all processed images
    images = [file for file in os.listdir(processed_folder) if file.endswith(".png")]
    
    if len(images) == 0:
        raise ValueError(f"No processed images found in {processed_folder}")
    
    if verbose:
        print(f"\nCreating training arrays...")
        print(f"Found {len(images)} processed images")
    
    image_ls = []
    segment_ls = []
    valid_paths = []
    
    iterator = tqdm(images, desc="Loading images and masks") if verbose else images
    
    for img_name in iterator:
        img_path = os.path.join(root_path, processed_folder, img_name)
        seg_path = os.path.join(root_path, segment_folder, img_name)
        
        if not os.path.exists(seg_path):
            if verbose:
                print(f"  WARNING: Segmentation not found for {img_name}, skipping")
            continue
        
        try:
            # Load image and segmentation mask
            image = ski.io.imread(img_path)
            segment = ski.io.imread(seg_path)
            
            # Normalize to [0, 1]
            image_ls.append(image / 255.0)
            
            # Ensure segmentation is single channel
            if segment.ndim == 2:
                segment = segment[:, :, np.newaxis]
            elif segment.ndim > 3:
                raise ValueError(f"Unexpected segmentation dimensions for {img_name}: {segment.shape}")
            segment_ls.append(segment[:, :, 0] / 255.0)
            
            valid_paths.append(img_name)
            
        except Exception as e:
            print(f"  ERROR loading {img_name}: {e}")
    
    if len(valid_paths) == 0:
        raise ValueError("No valid image-segmentation pairs found")
    
    # Convert to numpy arrays
    path_arr = np.asarray(valid_paths)
    image_arr = np.asarray(image_ls)
    segment_arr = np.asarray(segment_ls)
    
    # Save arrays
    output_dir = os.path.join(root_path, "training", experiment_name)
    path_file = os.path.join(output_dir, f"forsegment_{experiment_name}_paths.npy")
    image_file = os.path.join(output_dir, f"forsegment_{experiment_name}_images.npy")
    segment_file = os.path.join(output_dir, f"forsegment_{experiment_name}_segments.npy")

    np.save(path_file, path_arr)
    np.save(image_file, image_arr)
    np.save(segment_file, segment_arr)
    
    if verbose:
        print(f"\nSuccessfully created training data:")
        print(f"  Number of samples: {len(valid_paths)}")
        print(f"  Image shape: {image_arr.shape}")
        print(f"  Segment shape: {segment_arr.shape}")
        print(f"\nSaved files:")
        print(f"  {path_file}")
        print(f"  {image_file}")
        print(f"  {segment_file}")
    
    return len(valid_paths), (path_file, image_file, segment_file)


def check_masks_exist(processed_folder, segment_folder):
    """
    Check if segmentation masks exist for all processed images.
    Returns (bool, list_of_missing_files)
    """
    images = [file for file in os.listdir(processed_folder) if file.endswith(".png")]
    missing = []
    for img in images:
        if not os.path.exists(os.path.join(segment_folder, img)):
            missing.append(img)
    return len(missing) == 0, missing


def main():
    """Main function to run the segmentation data generation pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate training data for wing segmentation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default dataset (data_droso)
  python generate_segmentation_data.py
  
  # Process specific dataset
  python generate_segmentation_data.py --experiment data_tsetse
  
  # Only process images (skip array creation)
  python generate_segmentation_data.py --process-only
  
  # Only create arrays (images already processed)
  python generate_segmentation_data.py --arrays-only
  
  # Use custom image size
  python generate_segmentation_data.py --image-size 640

For more information, see the README.md in this folder.
        """
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        default="data_droso",
        help="Name of the experiment/dataset folder (default: data_droso)"
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
        "--process-only",
        action="store_true",
        help="Only process images, don't create training arrays"
    )
    
    parser.add_argument(
        "--arrays-only",
        action="store_true",
        help="Only create training arrays, skip image processing"
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

    processed_folder = os.path.join(root_path, "training", args.experiment, "segmentation", "image_processed")
    segment_folder = os.path.join(root_path, "training", args.experiment, "segmentation", "segmentation_fromannotator")

    if verbose:
        print("=" * 70)
        print("ITHILDIN Wing Analysis - Segmentation Training Data Generation")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Experiment: {args.experiment}")
        print(f"  Root path: {root_path}")
        print(f"  Image size: {image_size} x {image_size // 2}")
        print(f"  Mode: {'Process only' if args.process_only else 'Arrays only' if args.arrays_only else 'Full pipeline'}")
        print()
    
    try:
        # Step 1: Process images
        if not args.arrays_only:
            if verbose:
                print("\n" + "=" * 70)
                print("STEP 1: Processing Images")
                print("=" * 70)
            
            num_processed = process_images(args.experiment, root_path, image_size, verbose)
            
            if num_processed == 0:
                print("ERROR: No images were processed successfully")
                return 1

        # Step 2: Create training arrays for images with masks only
        if not args.process_only:
            # Check which processed images are missing masks
            masks_ok, missing = check_masks_exist(processed_folder, segment_folder)
            if not masks_ok:
                print("\n" + "=" * 70)
                print("WARNING: Some segmentation masks are missing.")
                print("=" * 70)
                print(f"Processed images found in: {processed_folder}")
                print(f"Missing segmentation masks for {len(missing)} images.")
                print("The training arrays will only include images with both a processed image and a corresponding mask.")
                print("To include all images, annotate the missing masks and re-run this script.")
                print("\nMissing files:")
                for f in missing[:10]:
                    print("  ", f)
                if len(missing) > 10:
                    print(f"  ...and {len(missing)-10} more.")
                print()

            if verbose:
                print("\n" + "=" * 70)
                print("STEP 2: Creating Training Arrays")
                print("=" * 70)
            
            num_samples, output_files = create_training_arrays(args.experiment, root_path, verbose)
            
            if num_samples == 0:
                print("ERROR: No training samples were created")
                return 1
        
        if verbose:
            print("\n" + "=" * 70)
            print("SUCCESS: Segmentation training data generation completed!")
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
