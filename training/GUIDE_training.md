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
