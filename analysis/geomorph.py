"""
R-based geometric morphometrics using geomorph package.

This module provides integration with R's geomorph package for:
- Procrustes superimposition
- Sliding semilandmark analysis
- TPS file I/O for coordinate data

Uses subprocess calls to execute R scripts for morphometric analyses
that are not easily replicated in Python.
"""

import os
import subprocess

import numpy as np
import pandas as pd

from config import CONFIG

def save_tps(dataframe, output_path, filenames=None):
    """
    Saves a list of landmark matrices into a .tps file.
    Optionally includes image filenames as IMAGE=.
    """

    x_cols = sorted([c for c in dataframe.columns if "X_" in c], 
                    key=lambda x: int(x.split('_')[-1]))
    y_cols = sorted([c for c in dataframe.columns if "Y_" in c], 
                    key=lambda x: int(x.split('_')[-1]))
    
    X = dataframe[x_cols].to_numpy()
    Y = dataframe[y_cols].to_numpy() 

    coords_arr = np.stack((X, Y), axis=2) # Shape: (samples, landmarks, dimensions)

    with open(output_path, 'w') as f:
        for i, landmarks in enumerate(coords_arr):
            landmarks = np.array(landmarks)
            p = landmarks.shape[0]
            f.write(f"LM={p}\n")

            for row in landmarks:
                f.write(f"{row[0]} {row[1]}\n")

            # Write IMAGE= if filenames provided
            if filenames is not None and i < len(filenames):
                f.write(f"ID={filenames[i].split(os.sep)[-1]}\n")
                f.write(f"IMAGE={filenames[i].split(os.sep)[-1]}\n")
            else:
                f.write(f"ID={i}\n\n")

            

def save_sliders(sliders_path, N_semi=CONFIG["semilandmarks_per_connection"], slm_p_connection=CONFIG["semilandmarks_per_connection"]):
    """
    Generates the 3-column matrix for geomorph from a list of curve paths.
    Each path is a list of R-indices (1-based).
    """
    # We build a list of lists, where each inner list is a full path of R-indices (1-based)
    curve_definitions = []
    current_semi_idx = CONFIG["N_landmarks"] + 1 # Semilandmarks start at R-index 18

    for (start, end), count in zip(CONFIG["allowed_connections"], slm_p_connection):
        # R-indices: start/end are fixed (1-17), semis are 18-47
        start_r = start + 1
        end_r = end + 1
        path_semis = list(range(current_semi_idx, current_semi_idx + count))
        
        # Create the sequence: [Fixed_Start, Semi1, Semi2, ..., Fixed_End]
        full_curve_path = [start_r] + path_semis + [end_r]
        curve_definitions.append(full_curve_path)
        
        # Increment global counter for semilandmarks
        current_semi_idx += count

    sliders = []
    for curve in curve_definitions:
        for i in range(1, len(curve) - 1):
            # triplet: [before, sliding_point, after]
            sliders.append([curve[i-1], curve[i], curve[i+1]])

    sliders = np.array(sliders)
    pd.DataFrame(sliders).to_csv(sliders_path, index=False, header=False)


def run_r_analysis(dataframe, semilandmark, script_name, N_semi=CONFIG["N_semilandmarks"], slm_p_connection=CONFIG["semilandmarks_per_connection"], filenames=None):
    
    input_path_csv = os.path.join(CONFIG["root_path"], "analysis", "temp", "input.csv")
    input_path_tps = os.path.join(CONFIG["root_path"], "analysis", "temp", "input.tps")
    sliders_path = os.path.join(CONFIG["root_path"], "analysis", "temp", "sliders.csv")
    output_path = os.path.join(CONFIG["root_path"], "analysis", "temp", "output.csv")
    script_path = os.path.join(CONFIG["root_path"], "analysis", script_name)

    # Prepare input files for R script
    if semilandmark:
        save_tps(dataframe, input_path_tps, filenames=filenames)
        save_sliders(sliders_path,  N_semi=N_semi, slm_p_connection=slm_p_connection)
        dataframe.to_csv(input_path_csv, index=False, sep=",")

    else:
        save_tps(dataframe, input_path_tps, filenames=filenames)
        dataframe.to_csv(input_path_csv, index=False, sep=",")

    # Run the R script
    try:
        result = subprocess.run(
            ["Rscript", script_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}!")
        print("Exit code:", e.returncode)
        print("Stdout:\n", e.stdout)
        print("Stderr:\n", e.stderr)
        raise

    # Load the output CSV
    return pd.read_csv(output_path, sep=",")

# Wrapper functions
def procrustes_analysis(dataframe, filenames=None):
    return run_r_analysis(dataframe, False, "procrustes.R", filenames=filenames)

def procrustes_semilandmark_analysis(dataframe, N_semi=CONFIG["N_semilandmarks"], slm_p_connection=CONFIG["semilandmarks_per_connection"], filenames=None):
    return run_r_analysis(dataframe, True, "procrustes_semilandmarks.R", N_semi=N_semi, slm_p_connection=slm_p_connection, filenames=filenames)

def anova_analysis(dataframe, filenames=None):
    return run_r_analysis(dataframe, False, "anova.R", filenames=filenames)

def anova_semilandmark_analysis(dataframe, N_semi=CONFIG["N_semilandmarks"], slm_p_connection=CONFIG["semilandmarks_per_connection"], filenames=None):
    return run_r_analysis(dataframe, True, "anova.R", N_semi=N_semi, slm_p_connection=slm_p_connection, filenames=filenames)
