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
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from analysis import geomorph
from transform import landmark_processing
from config import CONFIG

# Suppress sklearn warnings for cleaner output
warnings.simplefilter("ignore")

# Per-process cache of loaded reference models: (family, semilandmark) -> dict
_reference_models = {}

def _coord_sort_key(col):
    """Sort coordinate columns to X_0, Y_0, X_1, Y_1, ... as geomorph expects."""
    if "_" in col:
        prefix, idx = col.split("_")[0], col.split("_")[-1]  # this excludes the "sm" in semilandmarks
        return (int(idx), 0 if prefix == "X" else 1)
    return (float('inf'), 0)

def ensure_reference_model(semilandmark=False, family="mosquito"):
    """
    Returns the pre-fitted reference model for a family, building it if missing.

    The model consists of a GPA consensus shape (CSV, read by the R scripts for
    OPA alignment) and a pickled dict with the fitted LDA and outlier thresholds
    (95th percentile of the reference specimens' Avg/Max Procrustes distances to
    the consensus). Artifacts live in analysis/models/<family>/ and are rebuilt
    when missing or older than the reference CSV; delete the directory to force
    a rebuild (e.g. after a major sklearn upgrade breaks the pickle).

    Returns:
        dict: {"lda", "avg_max", "max_max", "consensus_path"}
    """
    cache_key = (family, semilandmark)
    if cache_key in _reference_models:
        return _reference_models[cache_key]

    kind = "semilandmarks" if semilandmark else "landmarks"
    model_dir = os.path.join(CONFIG["root_path"], "analysis", "models", family)
    consensus_path = os.path.join(model_dir, f"consensus_{kind}.csv")
    model_path = os.path.join(model_dir, f"reference_model_{kind}.pkl")

    reference_key = "semilandmark_reference_path" if semilandmark else "landmark_reference_path"
    reference_path = os.path.join(CONFIG["root_path"], CONFIG[reference_key])

    # Load existing artifacts unless the reference CSV is newer
    if os.path.exists(consensus_path) and os.path.exists(model_path):
        if os.path.getmtime(model_path) >= os.path.getmtime(reference_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            model["consensus_path"] = consensus_path
            _reference_models[cache_key] = model
            return model

    # Build step 1: GPA over the reference set to obtain the consensus shape
    # (mshape = mean of the GPA-aligned coordinates)
    reference_df = pd.read_csv(reference_path)
    reference_df_arr = reference_df[[col for col in reference_df.columns if "X_" in col or "Y_" in col]]
    reference_df_arr = reference_df_arr[sorted(reference_df_arr.columns, key=_coord_sort_key)]

    if semilandmark:
        gpa_df = geomorph.procrustes_semilandmark_analysis(reference_df_arr, family=family)
    else:
        gpa_df = geomorph.procrustes_analysis(reference_df_arr, family=family)

    coord_cols = [col for col in gpa_df.columns if ".X" in col or ".Y" in col]
    consensus = gpa_df[coord_cols].values.mean(axis=0).reshape(-1, 2)

    os.makedirs(model_dir, exist_ok=True)
    # Write + rename so concurrent workers never read a partial file
    pd.DataFrame(consensus, columns=["X", "Y"]).to_csv(consensus_path + ".tmp", index=False)
    os.replace(consensus_path + ".tmp", consensus_path)

    # Build step 2: re-align the raw reference to the consensus by OPA — the
    # exact transformation applied at inference (notably without semilandmark
    # sliding) — and fit the LDA and outlier thresholds on those coordinates
    if semilandmark:
        proc_df = geomorph.procrustes_semilandmark_analysis(reference_df_arr, family=family, consensus_path=consensus_path)
    else:
        proc_df = geomorph.procrustes_analysis(reference_df_arr, family=family, consensus_path=consensus_path)

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(proc_df[coord_cols].values, np.array(reference_df["TAXA LABEL"]))

    model = {
        "lda": lda,
        "avg_max": float(np.percentile(proc_df["Avg_Procrustes_Dist"], 95)),
        "max_max": float(np.percentile(proc_df["Max_Procrustes_Dist"], 95)),
    }
    with open(model_path + ".tmp", "wb") as f:
        pickle.dump(model, f)
    os.replace(model_path + ".tmp", model_path)

    model["consensus_path"] = consensus_path
    _reference_models[cache_key] = model
    return model

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

def procrustes(dataframe, semilandmark=False, N_semi=CONFIG["N_semilandmarks"], slm_p_connection=CONFIG["semilandmarks_per_connection"], family="mosquitoes"):
    
    prediction_df = dataframe
    prediction_df_arr = prediction_df[[col for col in prediction_df.columns if "X_" in col or "Y_" in col]]

    # Sort columns to fit geomorph
    prediction_df_arr = prediction_df_arr[sorted(prediction_df_arr.columns, key=_coord_sort_key)]
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
            filenames=filenames,
            family=family
        )
    else:
        prediction_proc_df = geomorph.procrustes_analysis(
            prediction_df_arr, 
            filenames=filenames,
            family=family
        )
        

    # Transfer the outlier detection
    prediction_df["Avg_Procrustes_Distance"] = prediction_proc_df["Avg_Procrustes_Dist"]
    prediction_df["Max_Procrustes_Distance"] = prediction_proc_df["Max_Procrustes_Dist"]
    prediction_proc_df.drop(columns="Avg_Procrustes_Dist", inplace=True)
    prediction_proc_df.drop(columns="Max_Procrustes_Dist", inplace=True)

    return prediction_df, prediction_proc_df

def procrustes_with_reference(dataframe, semilandmark=False, family="mosquito"):
    """
    Aligns new specimens to the stored reference consensus shape via ordinary
    Procrustes superimposition (OPA), building the reference model first if needed.
    """
    model = ensure_reference_model(semilandmark=semilandmark, family=family)

    # Split Dataframe into coordinates and non coordinates
    prediction_df_arr = dataframe[[col for col in dataframe.columns if "X_" in col or "Y_" in col]]
    prediction_df = dataframe[[col for col in dataframe.columns if "X_" not in col and "Y_" not in col]]

    # Save filename for tps file
    filenames = dataframe["File"].tolist()

    # Sort columns to fit geomorph
    prediction_df_arr = prediction_df_arr[sorted(prediction_df_arr.columns, key=_coord_sort_key)]

    # Align to the stored consensus (OPA instead of a full GPA)
    if semilandmark:
        prediction_proc_df = geomorph.procrustes_semilandmark_analysis(
            prediction_df_arr, filenames=filenames, family=family, consensus_path=model["consensus_path"])
    else:
        prediction_proc_df = geomorph.procrustes_analysis(
            prediction_df_arr, filenames=filenames, family=family, consensus_path=model["consensus_path"])

    return prediction_df, prediction_proc_df

def LDA(prediction_df, prediction_proc_df, semilandmark=False, family="mosquito"):

    # Copy to avoid SettingWithCopyWarning
    prediction_df = prediction_df.copy()

    X_predict = prediction_proc_df[[col for col in prediction_proc_df.columns if ".X" in col or ".Y" in col]].values

    # Use the pre-trained LDA model of the reference dataset
    lda = ensure_reference_model(semilandmark=semilandmark, family=family)["lda"]

    # Use model to predict new data
    Y_predict_scores = lda.predict_proba(X_predict)
    Y_predict_labels = lda.classes_

    # Parse predictions for json
    Y_predict_label = lda.predict(X_predict)
    Y_predict_score = np.max(Y_predict_scores, axis=1)

    print(Y_predict_label)

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