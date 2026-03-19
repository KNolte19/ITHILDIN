"""
Model prediction module for ITHILDIN wing analysis.

This module handles inference for all three main models:
- Segmentation: Wing boundary detection using U-Net++
- Landmark detection: Precise anatomical landmark localization
- Classification: Species identification

Models are loaded dynamically based on the selected insect family configuration.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF

from config_loader import get_config
from predictor import classification, landmark, segmentation

# Cache for loaded models to avoid reloading
_model_cache = {}


def get_models(family="mosquito"):
    """
    Get or load models for the specified insect family.
    
    Models are cached to avoid reloading on subsequent calls.
    
    Args:
        family (str): Insect family name
    
    Returns:
        tuple: (segment_model, landmark_model, classification_model, device)
    """
    if family not in _model_cache:
        config = get_config(family)
        device = config["device"]
        
        # Load models
        segment_model = segmentation.get_model(family).to(device)
        landmark_model = landmark.get_model(family).to(device)
        classification_model = classification.get_model(family).to(device) if config["has_classification"] else None
        
        _model_cache[family] = (segment_model, landmark_model, classification_model, device)
    
    return _model_cache[family]


def run_segmentation(
    image: np.ndarray, family="mosquito"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run segmentation inference to detect wing boundaries.

    Args:
        image (np.ndarray): Input image in HWC format with float32 type
        family (str): Insect family name

    Returns:
        tuple: Two arrays:
            - Sigmoid-activated segmentation mask (np.ndarray): Binary prediction (0-1)
            - Raw model prediction logits (np.ndarray): Unnormalized output
    """
    segment_model, _, _, device = get_models(family)
    tensor_image = TF.to_tensor(image.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        segment_model.eval()
        prediction = segment_model(tensor_image)

    prediction = prediction.squeeze().cpu().numpy()
    sigmoid_prediction = torch.sigmoid(torch.tensor(prediction)).numpy()
    return sigmoid_prediction, prediction


def run_landmark_detection(
    image: np.ndarray, segmentation_mask: np.ndarray, family="mosquito"
) -> np.ndarray:
    """
    Predict anatomical landmark positions on wing image.

    Uses both the original image and segmentation mask as inputs to
    improve landmark localization accuracy.

    Args:
        image (np.ndarray): Input image in HWC format
        segmentation_mask (np.ndarray): Predicted segmentation mask (H x W)
        family (str): Insect family name

    Returns:
        np.ndarray: Predicted landmark heatmaps of shape (N_landmarks, H, W)
    """
    _, landmark_model, _, device = get_models(family)
    tensor_image = TF.to_tensor(image.astype(np.float32)).unsqueeze(0).to(device)
    tensor_segment = (
        TF.to_tensor(segmentation_mask.astype(np.float32)).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        landmark_model.eval()
        prediction = landmark_model(tensor_image, tensor_segment)

    return torch.sigmoid(prediction).squeeze().cpu().numpy()


def run_classification(image: np.ndarray, family="mosquito", calibration=False) -> np.ndarray:
    """
    Classify wing image to predict species/taxa.
    
    Note: CNN classification is only available for mosquito family.

    Args:
        image (np.ndarray): Input image in HWC format
        family (str): Insect family name

    Returns:
        np.ndarray: Class probabilities for each species (shape: N_classes)
                   Values are in range [0, 1] after sigmoid activation
                   Returns None if classification is not available for the family
    """
    config = get_config(family)
    if not config["has_classification"]:
        return None
    
    _, _, classification_model, device = get_models(family)
    tensor_image = TF.to_tensor(image.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        classification_model.eval()
        prediction = classification_model(tensor_image)
        prediction_prob = prediction.cpu().detach().numpy() #TODO: Replace Sigmoid with Calibrated Probabilities

    if calibration:
        prediction_prob = calibrate_classification(prediction_prob.flatten())

    return prediction_prob

def calibrate_classification(prediction_logits: np.ndarray, coef=1.7832, intercept=-6.1032) -> np.ndarray:
    """
    Calibrate classification logits to probabilities using calibrated logistic function.

    Args:
        prediction_logits (np.ndarray): Raw model output logits (shape: N_classes)

    Returns:
        np.ndarray: Calibrated class probabilities (shape: N_classes)
    """

    def logistic_calibrator(score, coef=coef, intercept=intercept):
        """
        Apply a logistic transform p = 1 / (1 + exp(-(intercept + coef * score))).

        - score: scalar or array-like of raw scores
        - coef, intercept: numeric parameters; defaults are the ones learned above

        Returns numpy array (or scalar float for scalar input).
        """
        s = np.asarray(score, dtype=float)
        logits = intercept + coef * s
        probs = 1.0 / (1.0 + np.exp(-logits))
        # return scalar for scalar input
        if probs.shape == ():
            return float(probs)
        return probs

    score_ls = []
    for score in prediction_logits:
        cal_score = logistic_calibrator(score, coef, intercept)
        score_ls.append(cal_score)

    return np.array(score_ls)
