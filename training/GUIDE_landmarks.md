# Landmark Training Data Preparation Checklist

Use this checklist to ensure you have everything ready before running the scripts. We have prepared a small sample dataset in `training/example/` for testing the training scripts before using your own data. The data generation scripts will create the necessary .npy files in the correct format. Make sure to run those scripts first and verify the output files are created in `training/example/` before proceeding with training. Please note that the sample dataset is very small and is only meant for testing the training pipeline, so don't expect good performance from models trained on this data. It's just to ensure that the training scripts are working correctly with the expected input format. If you want to then train. on your own data use can use the same data scheme as the example dataset, but replace the images and annotations with your own.

## For Landmark Training Data

### Before You Start

- [ ] I have raw wing images in a supported format (JPG, PNG, TIFF)
- [ ] I have a CSV file with landmark annotations (see below)
- [ ] I know how many landmarks per image 
- [ ] I have chosen a name for my experiment (e.g., "example")
- [ ] I have trained the segmentation model on my data

### CSV File Requirements

- [ ] CSV has column: `filename` (e.g., "image001.jpg")
- [ ] CSV has coordinate columns starting from 0 in pixel values: `X_0, Y_0, X_1, Y_1, ...`
- [ ] Number of X/Y pairs matches my landmark count
- [ ] All filenames in CSV match actual image files

#### Example CSV Format
```csv
filename,X_0,Y_0,X_1,Y_1,X_2,Y_2,
image001.jpg,1013,207,749,319,255,626,
image002.jpg,902,198,625,322,139,667,
```

### Folder Structure Setup

- [ ] Created folder: `training/example/landmark/image_raw/`
- [ ] Copied all raw images to `image_raw/`
- [ ] Placed annotations CSV at: `training/example/landmark/landmark_annotations.csv`
- [ ] Verified filenames match between folder and CSV

#### Example Structure
```
training/
  └── example/
      └── landmark/
          ├── image_raw/
          │   ├── image001.jpg
          │   └── ...
          └── landmark_annotations.csv
```

### Step 1: Process Raw Images

- [ ] Run the image processing script to align and normalize images:
    ```bash
    python training/generate_landmark_data.py --experiment example --n-landmarks 15 --process-only
    ```

- [ ] Confirm that `training/example/landmark/image_processed/` contains processed images

### Step 2: Generate Landmark Heatmaps

- [ ] The script above will also generate heatmaps for each image in `training/example/landmark/landmark_heatmaps/`
- [ ] For each image, you should see `{name}_map.npy` and `{name}_coords.npy`

### Step 3: Create Training Arrays

- [ ] Run the script to generate the .npy arrays for training:
    ```bash
    python training/generate_landmark_data.py --experiment example --n-landmarks 15 --arrays-only
    ```

- [ ] Confirm that the following files exist in `training/example/`:
  - [ ] `forlandmark_example_paths.npy`
  - [ ] `forlandmark_example_images.npy`
  - [ ] `forlandmark_example_heatmap.npy`
- [ ] Loaded arrays in Python to verify shapes are correct
- [ ] Visually inspected a few samples to ensure quality

---

## Common Issues Checklist

If the script reports errors, check:

### File Not Found Errors
- [ ] All paths are spelled correctly
- [ ] Folders exist in the right locations
- [ ] No typos in experiment name
- [ ] Files have correct extensions

### Missing Annotations
- [ ] Every image has an entry in the CSV
- [ ] Filenames match exactly (including extension)
- [ ] No extra spaces in filenames

### Format Errors
- [ ] Images are in supported formats
- [ ] CSV is properly formatted (no missing commas), file should be semicolon `;` separated and not tab or comma separated
- [ ] Coordinate columns are named correctly

### Memory Errors
- [ ] Not processing too many images at once
- [ ] Images aren't excessively large
- [ ] Have at least 8GB RAM available

---

## Train the Landmark Model
1. Update the script to use your experiment name and provide the path to the segmentation model
2. Run the appropriate training script or use the colab link:
    ```bash
    python training/train_landmark_model_local.py
    ```
Colab Links:
- [Train Landmark Model](https://drive.google.com/drive/folders/1_rAI4mhnU5WG1cFDbFC4SGefEG_6-bmL?usp=share_link)

---
