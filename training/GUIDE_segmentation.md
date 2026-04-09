# Segmentation Training Data Preparation Checklist

Use this checklist to ensure you have everything ready before running the scripts. We have prepared a small sample dataset in `training/example/` for testing the training scripts before using your own data. The data generation scripts will create the necessary .npy files in the correct format. Make sure to run those scripts first and verify the output files are created in `training/example/` before proceeding with training. Please note that the sample dataset is very small and is only meant for testing the training pipeline, so don't expect good performance from models trained on this data. It's just to ensure that the training scripts are working correctly with the expected input format. If you want to then train. on your own data use can use the same data scheme as the example dataset, but replace the images and annotations with your own.

## For Segmentation Training Data

### Before You Start

- [ ] I have raw wing images in a supported format (JPG, PNG, TIFF)
- [ ] I have chosen a name for my experiment (in this example case we use "example")

### Folder Structure Setup

- [ ] Created folder: `training/{experiment}/segmentation/image_raw/`
- [ ] Copied all raw images to `image_raw/`

### Step 1: Process Raw Images

- [ ] Run the image processing script to align and normalize images:
    ```bash
    cd training
    python generate_segmentation_data.py --experiment example --process-only
    ```
- [ ] Confirm that `training/{experiment}/segmentation/image_processed/` contains processed images

### Step 2: Annotate Processed Images

- [ ] Use an annotation tool (e.g., **Label Studio**, **napari-convpaint**) to create segmentation masks for each processed image in `image_processed/`
- [ ] Save the segmentation masks as PNG files with the **same filenames** as the processed images
- [ ] Place the masks in: `training/{experiment}/segmentation/segmentation_fromannotator/`
- [ ] Verify that each processed image has a corresponding mask

### Example Structure
```
training/
  └── my_dataset/
      └── segmentation/
          ├── image_raw/
          │   ├── image001.jpg
          │   ├── image002.png
          │   └── ...
          ├── image_processed/
          │   ├── image001.png
          │   ├── image002.png
          │   └── ...
          └── segmentation_fromannotator/
              ├── image001.png
              ├── image002.png
              └── ...
```

### Step 3: Create Training Arrays

- [ ] Run the script to generate the .npy arrays for training:
    ```bash
    python generate_segmentation_data.py --experiment my_dataset --arrays-only
    ```
- [ ] Confirm that the following files exist in `training/{experiment}/`:
  - [ ] `path_forsegment.npy`
  - [ ] `image_forsegment.npy`
  - [ ] `segment_forsegment.npy`
- [ ] Loaded arrays in Python to verify shapes are correct
- [ ] Visually inspected a few samples to ensure quality

---

## Ready to Train the Segmentation Model?

Next steps:
1. Navigate to the `training/` folder
2. Open the appropriate training notebook or use the colab links:
   - `train_segment_model.ipynb` for segmentation
   - `train_landmark_model.ipynb` for landmarks
3. Update the notebook to use your experiment name
4. Run the training!

Colab Links:
- [Train Segmentation Model](https://drive.google.com/drive/folders/1_rAI4mhnU5WG1cFDbFC4SGefEG_6-bmL?usp=share_link)

---
