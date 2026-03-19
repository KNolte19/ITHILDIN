"""
Landmark analysis and geometric morphometrics for ITHILDIN.

This module provides functions for:
- Centroid size calculation
- Wing orientation determination
- Procrustes analysis (with and without semilandmarks)
- Linear Discriminant Analysis (LDA) for species prediction

COORDINATE CONVENTION:
All coordinate arrays use (2, N) format where:
- Row 0 = X coordinates (horizontal axis)
- Row 1 = Y coordinates (vertical axis)

This convention is maintained throughout the analysis pipeline.

Integrates with R's geomorph package for advanced morphometric analyses.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from analysis import geomorph
from transform import landmark_processing
from config import CONFIG

# Suppress sklearn warnings for cleaner output
warnings.simplefilter("ignore") 

def centroid_size(coords):
    """
    Calculates the centroid size from a coordinate array.
    
    Parameters:
        coords (np.ndarray): Shape (2, N) array with coordinates.
                             Row 0 = X coordinates (horizontal axis).
                             Row 1 = Y coordinates (vertical axis).
        
    Returns:
        float: The centroid size (square root of sum of squared distances from centroid).
    """
    
    # Convert to shape (N, 2) for easier processing
    # coords.T gives us (N, 2) where each row is [x, y]
    coords_T = coords.T
    
    # Calculate centroid (mean position)
    centroid = np.mean(coords_T, axis=0)
    
    # Calculate sum of squared distances from centroid
    diffs = coords_T - centroid
    squared_dists = np.sum(diffs**2, axis=1)
    centroid_size = np.sqrt(np.sum(squared_dists))
    
    return centroid_size

def orientation(coords):
    """
    Determines the wing orientation from a coordinate array.
    
    Parameters:
        coords (np.ndarray): Shape (2, N) array with coordinates.
                             Row 0 = X coordinates (horizontal axis).
                             Row 1 = Y coordinates (vertical axis).
        
    Returns:
        string: Wing orientation ("left" or "right").
                "left" means the wing tip points to the left.
                "right" means the wing tip points to the right.
    """
    # coords[0] contains all X coordinates
    # Get X coordinates of specific landmarks for orientation determination
    left_x_coord = coords[0][CONFIG["index_most_left_landmark"]]
    right_x_coord = coords[0][CONFIG["index_most_right_landmark"]]

    # If the "left" landmark has a larger X value than the "right" landmark,
    # the wing is oriented to the left
    if left_x_coord > right_x_coord:
        orientation = "left"
    else:
        orientation = "right"

    return orientation

def procrustes(dataframe, semilandmark=False, N_semi=CONFIG["N_semilandmarks"], slm_p_connection=CONFIG["semilandmarks_per_connection"]):
    
    prediction_df = dataframe
    prediction_df_arr = prediction_df[[col for col in prediction_df.columns if "X" in col or "Y" in col]]

    # Extract the numbers from the column names and sort accordingly to fit geomorph
    def sort_key(col):
        if "_" in col:
            prefix, idx = col.split("_")[0], col.split("_")[-1] # this excludes the "sm" in semilandmarks
            return (int(idx), 0 if prefix == "X" else 1)
        return (float('inf'), 0)

    prediction_df_arr = prediction_df_arr[sorted(prediction_df_arr.columns, key=sort_key)]
    prediction_df = prediction_df[[col for col in prediction_df.columns if "X" not in col and "Y" not in col]]

    # Extract filenames if present
    filenames = None
    if "File" in dataframe.columns:
        filenames = dataframe["File"].tolist()

    # Do Procrustes Analysis 
    if semilandmark:
        prediction_proc_df = geomorph.procrustes_semilandmark_analysis(
            prediction_df_arr, 
            N_semi=N_semi, 
            slm_p_connection=slm_p_connection, 
            filenames=filenames
        )
    else:
        prediction_proc_df = geomorph.procrustes_analysis(
            prediction_df_arr, 
            filenames=filenames
        )

    # Transfer the outlier detection
    prediction_df["Avg_Procrustes_Distance"] = prediction_proc_df["Avg_Procrustes_Dist"]
    prediction_df["Max_Procrustes_Distance"] = prediction_proc_df["Max_Procrustes_Dist"]
    prediction_proc_df.drop(columns="Avg_Procrustes_Dist", inplace=True)
    prediction_proc_df.drop(columns="Max_Procrustes_Dist", inplace=True)

    return prediction_df, prediction_proc_df

def procrustes_with_reference(dataframe, semilandmark=False):
    # Load and process reference data
    if semilandmark: 
        reference_df = pd.read_csv(os.path.join(CONFIG["root_path"], CONFIG["semilandmark_reference_path"]))
    else: 
        reference_df = pd.read_csv(os.path.join(CONFIG["root_path"], CONFIG["landmark_reference_path"]))

    # Split Dataframes into coordinates and non coordiantes
    reference_df_arr = reference_df[[col for col in reference_df.columns if "X_" in col or "Y_" in col]]
    reference_df = reference_df[[col for col in reference_df.columns if "X_" not in col and "Y_" not in col]]
    
    prediction_df_arr = dataframe[[col for col in dataframe.columns if "X_" in col or "Y_" in col]]
    prediction_df = dataframe[[col for col in dataframe.columns if "X_" not in col and "Y_" not in col]]

    # Save filename for tps file
    filenames = dataframe["File"].tolist()

    # Concat new data to reference data
    concat_df_arr = pd.concat([reference_df_arr, prediction_df_arr])
    concat_df = pd.concat([reference_df, prediction_df])
    
    # Extract the numbers from the column names and sort accordingly to fit geomorph
    def sort_key(col):
        if "_" in col:
            prefix, idx = col.split("_")[0], col.split("_")[-1] # this excludes the "sm" in semilandmarks
            return (int(idx), 0 if prefix == "X" else 1)
        return (float('inf'), 0)

    concat_df_arr = concat_df_arr[sorted(concat_df_arr.columns, key=sort_key)]

    # Do Procrustes Analysis
    if semilandmark:
        proc_df = geomorph.procrustes_semilandmark_analysis(concat_df_arr, filenames=filenames)
    else:
        proc_df = geomorph.procrustes_analysis(concat_df_arr, filenames=filenames)

    # Split proc_df into reference and prediction based on the original dataframe sizes
    n_reference = reference_df_arr.shape[0]
    reference_proc_df = proc_df.iloc[:n_reference].reset_index(drop=True)
    prediction_proc_df = proc_df.iloc[n_reference:].reset_index(drop=True)

    return reference_df, reference_proc_df, prediction_df, prediction_proc_df

def LDA(reference_df, reference_proc_df, prediction_df, prediction_proc_df, target="TAXA LABEL", semilandmark = False):

    # Copy to avoid SettingWithCopyWarning
    prediction_df = prediction_df.copy()

    # Create features and labels
    X = reference_proc_df[[col for col in reference_proc_df.columns if ".X" in col or ".Y" in col]].values
    y = np.array(reference_df[target])

    X_predict = prediction_proc_df[[col for col in prediction_proc_df.columns if ".X" in col or ".Y" in col]].values

    # Train LDA model on reference data 
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X, y)

    # Use model to predict new data
    Y_predict_scores = lda.predict_proba(X_predict)
    Y_predict_labels = lda.classes_

    # Parse predictions for json
    Y_predict_label = lda.predict(X_predict)
    Y_predict_score = np.max(Y_predict_scores, axis=1)

    predictions_lda_map = {}
    for i, file in enumerate(prediction_df["File"].values):
        predictions_lda_map[file.split(os.sep)[-1]] = (Y_predict_labels, Y_predict_scores[i])

    # Assign predictions to prediction_df
    prediction_df.loc[:, "LDA Score"] = Y_predict_score
    prediction_df.loc[:, "LDA Prediction"] = Y_predict_label

    return prediction_df, prediction_proc_df, predictions_lda_map

def ANOVA(dataframe, targets=None, semilandmark=False):
    if type(targets) == str:
        targets = [targets]

    if targets != None:
        prediction_df, prediction_proc_df = procrustes(dataframe, semilandmark=semilandmark)

        for target in targets:
            prediction_proc_df[target.lower()] = list(prediction_df[target])
        
        results = geomorph.anova_analysis(prediction_proc_df)
    else:
        raise Exception("No target variables given, please state at least one independent variable")

    return results

def detect_outlier(dataframe, proc_dataframe, max_max="default", avg_max="default"):
    """
    Detects outliers in the dataframe based on Procrustes distances.
    Adds an 'Outlier' column with True (outlier) or False (not outlier).
    Returns the updated dataframe and proc_dataframe.
    """
    dataframe = dataframe.copy()
    proc_dataframe = proc_dataframe.copy()

    dataframe["Max_Procrustes_Distance"] = proc_dataframe["Max_Procrustes_Dist"].copy()
    dataframe["Avg_Procrustes_Distance"] = proc_dataframe["Avg_Procrustes_Dist"].copy()
    dataframe["Outlier"] = False

    try:
        if max_max == "default":
            max_max = np.percentile(np.asarray(dataframe["Max_Procrustes_Distance"].dropna()), 95)
        if avg_max == "default":
            avg_max = np.percentile(np.asarray(dataframe["Avg_Procrustes_Distance"].dropna()), 95)

        mask = (dataframe["Avg_Procrustes_Distance"] > avg_max) & (dataframe["Max_Procrustes_Distance"] > max_max)
        dataframe.loc[mask, "Outlier"] = True
    except Exception:
        print("Outlier Detection not possible")

    return dataframe