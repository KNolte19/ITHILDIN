# Landmark Training Data Preparation Checklist

Use this checklist to ensure you have everything ready before running the scripts.

## For Landmark Training Data

### Before You Start

- [ ] I have raw wing images in a supported format (JPG, PNG, TIFF)
- [ ] I have a CSV file with landmark annotations (see below)
- [ ] I know how many landmarks per image (e.g., 11 or 14)
- [ ] I have chosen a name for my experiment (e.g., "my_dataset")

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

- [ ] Created folder: `training/{experiment}/landmark/image_raw/`
- [ ] Copied all raw images to `image_raw/`
- [ ] Placed annotations CSV at: `training/{experiment}/landmark/landmark_annotations.csv`
- [ ] Verified filenames match between folder and CSV

#### Example Structure
```
training/
  └── my_dataset/
      └── landmark/
          ├── image_raw/
          │   ├── image001.jpg
          │   └── ...
          └── landmark_annotations.csv
```

### Step 1: Process Raw Images

- [ ] Run the image processing script to align and normalize images:
    ```bash
    python training/generate_landmark_data.py --experiment my_dataset --n-landmarks 11 --process-only
    ```
    *(Replace `11` with your actual number of landmarks)*

- [ ] Confirm that `training/{experiment}/landmark/image_processed/` contains processed images

### Step 2: Generate Landmark Heatmaps

- [ ] The script above will also generate heatmaps for each image in `landmark/landmark_heatmaps/`
- [ ] For each image, you should see `{name}_map.npy` and `{name}_coords.npy`

### Step 3: Create Training Arrays

- [ ] Run the script to generate the .npy arrays for training:
    ```bash
    python generate_landmark_data.py --experiment my_dataset --n-landmarks 11 --arrays-only
    ```
    *(Replace `11` with your actual number of landmarks)*

- [ ] Confirm that the following files exist in `training/{experiment}/`:
  - [ ] `forlandmark_{experiment}_paths.npy`
  - [ ] `forlandmark_{experiment}_images.npy`
  - [ ] `forlandmark_{experiment}_heatmap.npy`
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
- [ ] CSV is properly formatted (no missing commas)
- [ ] Coordinate columns are named correctly

### Memory Errors
- [ ] Not processing too many images at once
- [ ] Images aren't excessively large
- [ ] Have at least 8GB RAM available

---

## Ready to Train the Landmark Model?
Next steps:
1. Navigate to the `training/` folder or use the colab links
2. Open the appropriate training notebook:
   - `train_landmark_model.ipynb` for landmarks
3. Update the notebook to use your experiment name and landmark count
4. Run the training!

Colab Links:
- [Train Segmentation Model](https://colab.research.google.com/drive/1-2-3-4-5-6-7-8-9-0)
- [Train Landmark Model](https://colab.research.google.com/drive/1-2-3-4-5-6-7-8-9-0)

---
