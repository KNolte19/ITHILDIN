"""
Main module for ITHILDIN wing analysis pipeline.

This module provides the core functionality for:
- Running predictions on wing images (segmentation, landmarks, classification)
- Processing landmark data and generating predictions
- Integrating multiple analysis components into a unified pipeline
"""

import json
import os
import time

import numpy as np
import pandas as pd

from analysis import landmark_analysis, geomorph
from config import update_config
from config_loader import get_config
from predictor import prediction
from transform import (
    image_processing,
    landmark_processing,
    segmentation_processing,
    wing_processing,
)
import utils


def run_prediction(file, save_path="test", family="mosquito", timing_info=None, stream=True, save_image=True, num_landmarks_ref="default"):
    """
    Run complete wing analysis pipeline on an input image.

    This function performs the full ITHILDIN analysis workflow including:
    1. Image preprocessing and alignment
    2. Wing segmentation
    3. Landmark detection
    4. Species classification (if available for the family)
    5. Skeleton extraction and repair
    6. Semilandmark generation
    7. Coordinate scaling and analysis

    Args:
        file: Input image file or file stream
        save_path (str): Path prefix for saving output files (without extension).
                        Default is "test".
        family (str): Insect family name ('mosquito', 'drosophila', or 'tsetse')
        timing_info (dict): Optional dictionary to store timing information for each step

    Returns:
        None. Saves results to JSON and PNG files at the specified path.

    Output Files:
        - <save_path>.json: Complete analysis results including coordinates and predictions
        - <save_path>.png: Visualization with landmarks and semilandmarks overlaid
    """
    if timing_info is None:
        timing_info = {}
    
    pipeline_start = time.time()
    
    # Update global CONFIG for this family (so transform modules use correct config)
    update_config(family)
    
    # Get configuration for specified family
    CONFIG = get_config(family)

    if num_landmarks_ref == "default":
        num_landmarks_ref = CONFIG["semilandmarks_per_connection"]
    
    # Define output paths
    json_save_path = f"{save_path}.json"
    image_save_path = f"{save_path}.png"

    # Step 1: Image preprocessing
    step_start = time.time()
    image_aligned, mask_aligned, mask = image_processing.process_image(
        file, from_stream=stream
    )

    image_ithildin = image_processing.transform_image(
        image_aligned,
        mask_aligned,
        contrast="Soft",
        resize_size=CONFIG["segmentation_image_size"][0],
    )
    timing_info["preprocessing"] = time.time() - step_start

    # Step 2: Segmentation prediction
    step_start = time.time()
    segmentation_sigmoid, segmentation_logit = prediction.run_segmentation(image_ithildin, family)
    timing_info["segmentation"] = time.time() - step_start

    # Step 3: Landmark prediction
    step_start = time.time()
    image_landmark = image_processing.resize(
        image_ithildin, CONFIG["landmark_image_size"][0]
    )
    segmentation_landmark = image_processing.resize(
        segmentation_sigmoid, CONFIG["landmark_image_size"][0]
    )

    landmarks = prediction.run_landmark_detection(image_landmark, segmentation_landmark, family)
    landmarks_coord = landmark_processing.heatmap_to_coordinates(landmarks)
    timing_info["landmark_detection"] = time.time() - step_start

    # Step 4: Species classification (only for families that support it)
    predicted_species = None
    predicted_species_proba = None
    if CONFIG["has_classification"]:
        step_start = time.time()
        image_classifier = image_processing.transform_image(
            image_aligned,
            mask_aligned,
            contrast="Strong",
            resize_size=CONFIG["classifier_image_size"][0],
        )
        predicted_species_scores = prediction.run_classification(image_classifier, family, calibration=True)

        # Parse predictions for json
        predicted_species = CONFIG["classifier_species_list"][np.argmax(predicted_species_scores)]
        predicted_species_proba = np.max(predicted_species_scores)
        
        predictions_cnn = [{"species": s, "score": float(p)} for s, p in zip(CONFIG["classifier_species_list"], predicted_species_scores)]
        predictions_cnn_map = dict(zip(CONFIG["classifier_species_list"], predicted_species_scores))
        
        timing_info["classification"] = time.time() - step_start

    # Step 5: Extract skeleton and junctions
    step_start = time.time()
    skeleton = segmentation_processing.skeletonize(segmentation_sigmoid)
    junctions_coord = segmentation_processing.extract_junction_coordinates(skeleton)
    consensus_coord = landmark_processing.consensus_coordinates(
        landmarks_coord, junctions_coord, skeleton
    )
    repaired_skeleton, status, failed_coord = wing_processing.check_skeleton(
        skeleton, consensus_coord, segmentation_logit
    )
    timing_info["skeleton_processing"] = time.time() - step_start

    # Step 6: Re-extract if skeleton was repaired
    step_start = time.time()
    if status == "Repaired":
        junctions_coord = segmentation_processing.extract_junction_coordinates(
            repaired_skeleton
        )
        consensus_coord = landmark_processing.consensus_coordinates(
            landmarks_coord, junctions_coord, repaired_skeleton
        )
        repaired_skeleton, _, failed_coord = wing_processing.check_skeleton(
            repaired_skeleton, consensus_coord, segmentation_logit
        )
    timing_info["skeleton_repair"] = time.time() - step_start

    # Step 7: Generate semilandmarks
    step_start = time.time()
    semi_landmarks = landmark_processing.create_semi_landmarks(
        consensus_coord, repaired_skeleton, num_landmarks_ref=num_landmarks_ref
    )

    # If semilandmark generation failed (often happens when two points are too close), fill with NaNs and update status
    if len(semi_landmarks[0]) != np.sum(num_landmarks_ref):
        semi_landmarks = np.full((np.sum(num_landmarks_ref), 2), np.nan)
        
    timing_info["semilandmark_generation"] = time.time() - step_start

    # Step 8: Scale coordinates to original image dimensions
    step_start = time.time()
    scaled_consensus_coord = landmark_processing.rescale_coordinates(
        consensus_coord, mask, mask_aligned
    )
    scaled_slm_coord = landmark_processing.rescale_coordinates(
        semi_landmarks, mask, mask_aligned
    )
    timing_info["coordinate_scaling"] = time.time() - step_start

    # Step 9: Calculate morphometric features
    step_start = time.time()
    centroid_size = landmark_analysis.centroid_size(scaled_consensus_coord)
    orientation = landmark_analysis.orientation(consensus_coord)
    timing_info["morphometric_analysis"] = time.time() - step_start

    # Prepare output data
    data = {
        "file_name": str(image_save_path),
        "status": str(status),
        "failed_coordinate": (
            None if failed_coord is None else np.array(failed_coord).tolist()
        ),
        "predicted_coords": np.array(landmarks_coord).tolist(),
        "junctions_coords": np.array(junctions_coord).tolist(),
        "landmark_coords": np.array(consensus_coord).tolist(),
        "semilandmark_coords": np.array(semi_landmarks).tolist(),
        "scaled_landmark_coords": np.array(scaled_consensus_coord).tolist(),
        "scaled_semilandmark_coords": np.array(scaled_slm_coord).tolist(),
        "scaled_centroid": float(centroid_size),
        "orientation": orientation,
        "cnn_prediction": {
            "top": str(predicted_species) if CONFIG["has_classification"] else None,
            "score": float(predicted_species_proba) if CONFIG["has_classification"] else None,
            "map": predictions_cnn_map if CONFIG["has_classification"] else None,
        },
    }

    # Save results to JSON
    step_start = time.time()
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Save visualization
    if save_image:
        utils.save_image_with_landmarks(
            image_aligned, image_save_path, scaled_consensus_coord, scaled_slm_coord
        )

    timing_info["file_saving"] = time.time() - step_start
    
    timing_info["total_time"] = time.time() - pipeline_start
    
    return timing_info


def get_landmark_predictions(dataframe, session, has_classifier=False):
    """
    Generate landmark-based species predictions using Linear Discriminant Analysis.

    This function applies LDA models to predict species based on landmark
    and semilandmark coordinates. It processes the dataframe in two stages:
    1. Base predictions using landmarks only (for all samples)
    2. Enhanced predictions using semilandmarks (for samples where available)

    Args:
        dataframe (pd.DataFrame): Input dataframe containing landmark coordinates
                                 and metadata. Must include columns with X_ and Y_
                                 prefixes for landmark coordinates, and optionally
                                 X_sm_ and Y_sm_ for semilandmarks.

    Returns:
        pd.DataFrame: Original dataframe augmented with two new columns:
                     - LM_Predicted_Taxa: Predicted species/taxa label
                     - LM_Predicted_Score: Prediction confidence score (0-1)
    """
    family = session["family"]
    CONFIG = get_config(family)

    # Adjust wing orientation for consistent analysis
    dataframe = landmark_processing.adjust_orientation(dataframe.copy())

    # Separate landmark-only data (excluding semilandmarks)
    dataframe_landmarks = dataframe.loc[:, ~dataframe.columns.str.contains(r"_sm_", regex=True)].copy()

    # Filter for complete semilandmark data (no missing values)
    dataframe_semilandmarks = dataframe.dropna(axis=0, how="any")

    # Run Procrustes analysis and detect outliers
    if has_classifier:
        reference_df, reference_proc_df, prediction_df, prediction_proc_df = landmark_analysis.procrustes_with_reference(dataframe_landmarks, semilandmark=False)
        reference_df_semi, reference_proc_df_semi, prediction_df_semi, prediction_proc_df_semi = landmark_analysis.procrustes_with_reference(dataframe_semilandmarks, semilandmark=True)
        
        dataframe_landmarks = landmark_analysis.detect_outlier(dataframe_landmarks, prediction_proc_df)

        if len(dataframe_semilandmarks) > 0:
            dataframe_semilandmarks = landmark_analysis.detect_outlier(dataframe_semilandmarks, prediction_proc_df_semi)

    # If no classifier is available for this family, return dataframe without prediction columns
    if has_classifier:
                # Run LDA on landmarks only
        prediction_df_landmarks, prediction_proc_df_landmarks, predictions_lda_map_landmarks = landmark_analysis.LDA(
            reference_df, reference_proc_df,
            prediction_df, prediction_proc_df,
            target="TAXA LABEL", semilandmark=False
        )

        # Populate base predictions from landmark-only analysis
        #dataframe["Outlier"] = dataframe_landmarks["Outlier"]

        # Enhance predictions with semilandmarks where available
        if len(dataframe_semilandmarks) > 0:

            prediction_df_semilandmarks, prediction_proc_df_semilandmarks, predictions_lda_map_semilandmarks = landmark_analysis.LDA(
                reference_df_semi, reference_proc_df_semi,
                prediction_df_semi, prediction_proc_df_semi,
                target="TAXA LABEL", semilandmark=True
            )

            # Update only rows with complete semilandmark data
            #dataframe.loc[dataframe_semilandmarks.index, "Outlier"] = dataframe_semilandmarks["Outlier"]

        # Write results to dataframe
        for file in prediction_df_landmarks["File_Name"]:
            json_path = os.path.join(session["request_path_processed"], file.split(".")[0] + ".json")
            
            if prediction_df_landmarks.loc[prediction_df_landmarks["File_Name"] == file]["Status"].values[0] != "Failed Repair":
                try:
                    file_lda_map = predictions_lda_map_semilandmarks[file]
                    semilandmark_used = True
                except KeyError:
                    file_lda_map = predictions_lda_map_landmarks[file]
                    semilandmark_used = False
            else:
                file_lda_map = predictions_lda_map_landmarks[file]
                semilandmark_used = False

            predicted_species = file_lda_map[0][np.argmax(file_lda_map[1])]
            predicted_species_proba = np.max(file_lda_map[1])
            predictions_lda_map = dict(zip(file_lda_map[0], file_lda_map[1]))

            # Add results to JSON file
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            data["landmark_prediction"] = {
                    "top": str(predicted_species),
                    "score": float(predicted_species_proba),
                    "map": predictions_lda_map,
                    "semilandmark": semilandmark_used
                }

            # Calulate ensemble prediction by combining CNN and LDA predictions
            cnn_map = data["cnn_prediction"]["map"]
            lda_map = data["landmark_prediction"]["map"]

            average_map = {}
            for species in CONFIG["classifier_species_list"]:
                cnn_score = cnn_map.get(species, 0)
                lda_score = lda_map.get(species, 0)
                average_map[species] = (cnn_score + lda_score) / 2

            average_predicted_species = max(average_map, key=average_map.get)
            average_predicted_score = average_map[average_predicted_species]

            data["ensemble_prediction"] = {
                "top": str(average_predicted_species),
                "score": float(average_predicted_score),
                "map": average_map
            }   

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    return dataframe, dataframe_landmarks, dataframe_semilandmarks


def prepare_download(prediction_df, prediction_lm_df, prediction_slm_df, has_classifier, session):
    CONFIG = get_config(session["family"])

    base_columns = [
        "File_Name",
        "File",
        "Status",
    ]

    if has_classifier:    
        base_columns.extend([
                "CNN_Predicted_Taxa",
                "CNN_Predicted_Score",
                "Centroid",
                "Orientation",
                "LM_Predicted_Taxa",
                "LM_Predicted_Score",
                "ENS_Predicted_Taxa",
                "ENS_Predicted_Score"
        ])

    # Save CSV files for download 
    prediction_df_path = os.path.join(session["request_path"], f"coordinates_{session['identifier']}.csv")
    output_columns = base_columns + [col for col in prediction_df.columns if col.startswith(("X_", "Y_"))]
    prediction_df[output_columns].to_csv(prediction_df_path, sep=";")

    # Save raw coordinates to TPS file for download
    prediction_lm_df_tpspath = os.path.join(session["request_path"], f"coordinates_{session['identifier']}.tps")
    prediction_slm_df_tpspath = os.path.join(session["request_path"], f"coordinates_{session['identifier']}_semi.tps")

    geomorph.save_tps(prediction_lm_df, prediction_lm_df_tpspath, filenames=prediction_df["File_Name"].tolist())
    geomorph.save_tps(prediction_slm_df, prediction_slm_df_tpspath, filenames=prediction_df["File_Name"].tolist())

    # Save sliders file for semilandmarks
    sliders_save_path = os.path.join(session["request_path"], f"sliders_{session['identifier']}.csv")
    geomorph.save_sliders(sliders_save_path,  N_semi=CONFIG["N_semilandmarks"], slm_p_connection=CONFIG["semilandmarks_per_connection"])