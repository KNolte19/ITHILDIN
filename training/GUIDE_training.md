# ITHILDIN Training Scripts User Guide

This guide explains how to use the improved training scripts for the ITHILDIN wing analysis models.

## Overview

We provide **four training scripts** in two versions each:

### Landmark Detection Model Training
- **`train_landmark_model_local.py`** - For local training with direct repository access
- **`train_landmark_model_colab.ipynb`** - For training on Google Colab with Google Drive

### Wing Segmentation Model Training
- **`train_segment_model_local.py`** - For local training with direct repository access
- **`train_segment_model_colab.ipynb`** - For training on Google Colab with Google Drive

## Quick Start Guides

**Prerequisites:**
- Python 3.8+
- Required packages: `pip install -r requirements.txt`

### Step 1: Prepare Your Data
- Follow the instructions in `GUIDE_landmarks.md` to prepare your landmark training data
- Follow the instructions in `GUIDE_segmentation.md` to prepare your segmentation training data

We provide a small sample dataset in `training/example/` for testing the training scripts before using your own data. The data generation scripts will create the necessary .npy files in the correct format. Make sure to run those scripts first and verify the output files are created in `training/example/` before proceeding with training. Please note that the sample dataset is very small and is only meant for testing the training pipeline, so don't expect good performance from models trained on this data. It's just to ensure that the training scripts are working correctly with the expected input format.

### Option 1: Local Training (Python Scripts)

1. **(Optional) Edit the configuration section** at the top of the scripts:
   - `train_landmark_model_local.py` for landmark model
   - `train_segment_model_local.py` for segmentation model

2. **Run the script:**
   ```bash
   cd training

   # For Segmentation Model
   python train_segment_model_local.py

   # For Landmark Model
   python train_landmark_model_local.py

   ```

3. **Monitor training** - Progress and metrics will be printed to console

4. **Find your models** in `./training/example/` directory

### Option 2: Google Colab Training (Notebooks)

**Prerequisites:**
- Google account with Google Colab access
- Training data uploaded to Google Drive
- GPU runtime enabled in Colab

**Steps:**

1. **Upload to Colab:**
   - Upload `train_landmark_model_colab.ipynb` or `train_segment_model_colab.ipynb` to Google Colab
   - Upload the .npy training data files generated from the data preparation scripts to your Google Drive

2. **Enable GPU:**
   - Go to: Runtime > Change runtime type
   - Set Hardware accelerator to: GPU
   - Recommended: T4 or better

3. **Edit Configuration:**
   - Find the "Configuration Parameters" section
   - Update the paths to your Google Drive location
   - Adjust other parameters as needed

4. **Run All Cells:**
   - Runtime > Run all
   - Or run cells one by one (Shift+Enter)

5. **Monitor Progress:**
   - Watch the training metrics in the output
   - Models are automatically saved to your specified Google Drive path

## Output Files

### Landmark Model Outputs

Located in `OUTPUT_DIR` (default: `./training/example/`):

- `landmark_fold-{FOLD}_{EXPERIMENT}.pth` - Full model (for inference)
- `landmark_weights_fold-{FOLD}_{EXPERIMENT}.pth` - Model weights only (for transfer learning)
- `test_files_fold-{FOLD}_{EXPERIMENT}.npy` - List of paths of test images
- `training_preview_fold{FOLD}.png` - Data preview (if enabled)
- `segmentation_preview_fold{FOLD}.png` - Segmentation preview (if enabled)

### Segmentation Model Outputs

Located in `OUTPUT_DIR` (default: `./training/example/`):

- `segmentation_fold-{FOLD}.pth` - Full model
- `segmentation_weights_fold-{FOLD}.pth` - Model weights only
- `evaluation_metrics_fold-{FOLD}.json` - Evaluation metrics
- `training_preview_fold{FOLD}.png` - Data preview (if enabled)
- `predictions_fold{FOLD}.png` - Prediction visualization (if enabled)

## Troubleshooting

### Common Issues

#### "CUDA out of memory"
**Solution**: Reduce `BATCH_SIZE` or `IMAGE_SIZE`
```python
BATCH_SIZE = 2  # Instead of 4 or 8
```

#### "FileNotFoundError: Input data not found"
**Solution**: 
1. Check your `DATA_PATH` is correct
2. Ensure you've run the data generation scripts first
3. Verify file names match your `EXPERIMENT` name

#### Training is very slow
**Solutions**:
- Ensure GPU is being used (check "Using cuda device" message)
- For Colab: Enable GPU runtime
- Increase `NUM_WORKERS` for local training
- Reduce `NUM_WORKERS` to 12 for Colab

#### "RuntimeError: CUDA error: device-side assert triggered"
**Solution**: This usually means data dimensions don't match. I would recommend you debug on cpu.
- Verify `N_LANDMARKS` matches your data
- Check `IMAGE_HEIGHT`, `IMAGE_WIDTH`, `IMAGE_SIZE` are correct

### Using Pretrained Weights

For landmark model:
```python
PRETRAINED_MODEL_PATH = "training/models_mosquito/mosquito_landmark_weights_fold-1.pth"
```

For segmentation model:
```python
PRETRAINED_BOOL = True
PRETRAINED_MODEL_PATH = "training/models_mosquito/mosquito_segmentation_weights_fold-1.pth"
```