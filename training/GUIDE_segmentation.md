# Segmentation Training Data Preparation Checklist

Use this checklist to ensure you have everything ready before running the scripts. We have prepared a small sample dataset in `training/example/` for testing the training scripts before using your own data. The data generation scripts will create the necessary .npy files in the correct format. Make sure to run those scripts first and verify the output files are created in `training/example/` before proceeding with training. Please note that the sample dataset is very small and is only meant for testing the training pipeline, so don't expect good performance from models trained on this data. It's just to ensure that the training scripts are working correctly with the expected input format. If you want to then train on your own data use can use the same data scheme as the example dataset, but replace the images and annotations with your own.

## For Segmentation Training Data

### Before You Start

- [ ] I have raw wing images in a supported format (JPG, PNG, TIFF)
- [ ] I have chosen a name for my experiment (in this example case we use "example")
- [ ] You have installed the required Python packages and set up your environment according to the README instructions

### Folder Structure Setup

- [ ] Created folder: `training/{experiment}/segmentation/image_raw/`
- [ ] Copied all raw images to `image_raw/`

### Step 1: Process Raw Images

- [ ] Run the image processing script to align and normalize images:
    ```bash
    python training/generate_segmentation_data.py --experiment example --process-only
    ```
- [ ] Confirm that `training/{experiment}/segmentation/image_processed/` contains processed images

### Step 2: Annotate Processed Images
You can skip this process for the example dataset since we have included sample segmentation masks, but for your own data you will need to create segmentation masks for each processed image.
- [ ] Use an annotation tool (e.g., **Label Studio**, **napari-convpaint**) to create segmentation masks for each processed image in `image_processed/`
- [ ] Save the segmentation masks as PNG files with the **same filenames** as the processed images
- [ ] Place the masks in: `training/{experiment}/segmentation/segmentation_fromannotator/`
- [ ] Verify that each processed image has a corresponding mask

### Example Structure
```
training/
  └── example/
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
    python training/generate_segmentation_data.py --experiment example --arrays-only
    ```
- [ ] Confirm that the following files exist in `training/{experiment}/`:
  - [ ] `path_forsegment.npy`
  - [ ] `image_forsegment.npy`
  - [ ] `segment_forsegment.npy`
- [ ] Loaded arrays in Python to verify shapes are correct
- [ ] Visually inspected a few samples to ensure quality

---

## Train the Segmentation Model
1. Update the script to use your experiment name if you are not using the example 
2. Run the appropriate training script or use the colab link:
    ```bash
    python training/train_segment_model_local.py
    ```

Alternatively you can also use the Colab notebook for training the segmentattion model. We provided the already generated arrays and pretrained models in the Google Drive folder, so you can directly run the training cells without having to prepare your own data first. Note that the dataset there are larger to give you a better idea of the training process and expected performance, but you can also replace those with your own data by following the same data scheme as the example dataset.

- [Train Segmentation Model](https://drive.google.com/drive/folders/1_rAI4mhnU5WG1cFDbFC4SGefEG_6-bmL?usp=share_link)

---
