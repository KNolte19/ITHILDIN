import numpy as np
import pandas as pd


def _optimal_rotation(shape, ref):
    """Rotate `shape` onto `ref` via the orthogonal Procrustes / Kabsch solution.
    Reflections are forbidden (det(R) forced to +1) since a mirrored landmark
    configuration is not a biologically valid shape."""
    U, _, Vt = np.linalg.svd(shape.T @ ref)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return shape @ R


def _modified_z(values, thresh=3.5):
    """Iglewicz & Hoaglin modified z-score: (x - median) / (1.4826 * MAD).
    Robust to the very outliers you're trying to find, unlike mean/SD.
    thresh=3.5 is the standard recommended cutoff for this statistic."""
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        return np.zeros_like(values), np.zeros(len(values), dtype=bool)
    z = 0.6745 * (values - med) / mad
    return z, np.abs(z) > thresh


def identify_outliers(df, n_landmarks=12, group_col=None,
                       shape_thresh=3.5, landmark_thresh=3.5, min_group_n=4):
    """
    Flag likely landmarking/model errors for an automated GM pipeline
    (no manual visual inspection required).

    Assumes the coordinate columns in `df` are ALREADY GPA-superimposed
    (translated, scaled to unit centroid size, rotated to a consensus).
    No full alignment is redone here -- only a lightweight refinement:
    each specimen is re-rotated onto a MEDIAN-based reference rather than
    trusting whatever reference your upstream GPA converged to. Standard
    GPA consensus is typically mean-based, and the mean is exactly what
    the outliers you're hunting can distort -- this step corrects for
    that cheaply, and doubles as a safety net if your upstream step
    didn't include a full rotational fit.

    Two independent checks, both robust (median/MAD-based, not mean/SD):

    - Outlier_Shape: overall Procrustes distance to the median reference
      is unusually large -> catches globally distorted configurations.
    - Outlier_Landmark: any SINGLE landmark deviates from its reference
      position by an unusual amount -> catches localized model errors
      (one misplaced/swapped point) that get diluted and hidden inside
      a whole-shape distance when there are many landmarks.

    group_col: optional column (e.g. species) to compute the reference
    shape and cutoffs WITHIN each group. Important for fine-grained
    interspecific work -- otherwise genuine between-species differences
    inflate the reference spread and either mask real errors or flag
    legitimate small-sample-species specimens as outliers.

    Nothing is deleted. Outlier_Landmark_IDs records which landmark(s)
    triggered, so you can log/audit flags even without visual review.
    """
    coord_cols = [f"{i}.{ax}" for i in range(1, n_landmarks + 1) for ax in ("X", "Y")]
    X = df[coord_cols].to_numpy().reshape(len(df), n_landmarks, 2)  # already GPA coords

    n = len(df)
    proc_dist = np.full(n, np.nan)
    outlier_shape = np.zeros(n, dtype=bool)
    outlier_landmark = np.zeros(n, dtype=bool)
    landmark_ids = [[] for _ in range(n)]

    group_vals = (df[group_col].astype(str).to_numpy()
                  if group_col is not None else np.full(n, "all", dtype=object))

    for g in np.unique(group_vals):
        pos = np.where(group_vals == g)[0]
        if len(pos) < min_group_n:
            print(f"[warn] group '{g}' has only {len(pos)} specimens "
                  f"(< {min_group_n}) -- skipping outlier detection for it.")
            continue

        shapes = X[pos]

        # refine reference from mean-based (upstream GPA) to median-based (robust),
        # re-rotating each specimen onto it -- two passes is enough since we're
        # correcting an existing near-consensus, not solving one from scratch
        ref = np.median(shapes, axis=0)
        aligned = np.array([_optimal_rotation(s, ref) for s in shapes])
        ref = np.median(aligned, axis=0)
        aligned = np.array([_optimal_rotation(s, ref) for s in aligned])

        # whole-shape Procrustes distance to reference
        d = np.sqrt(((aligned - ref[None, :, :]) ** 2).sum(axis=(1, 2)))
        _, flag_shape = _modified_z(d, shape_thresh)

        # per-landmark deviation from reference position
        dev = np.linalg.norm(aligned - ref[None, :, :], axis=2)  # (n_specimens, n_landmarks)
        flag_landmark_group = np.zeros(len(shapes), dtype=bool)
        for lm in range(n_landmarks):
            _, flags_lm = _modified_z(dev[:, lm], landmark_thresh)
            flag_landmark_group |= flags_lm
            for i in np.where(flags_lm)[0]:
                landmark_ids[pos[i]].append(lm + 1)  # 1-indexed to match "1.X"/"1.Y" columns

        proc_dist[pos] = d
        outlier_shape[pos] = flag_shape
        outlier_landmark[pos] = flag_landmark_group

    df = df.copy()
    df["ProcrustesDist"] = proc_dist
    df["Outlier_Shape"] = outlier_shape
    df["Outlier_Landmark"] = outlier_landmark
    df["Outlier_Landmark_IDs"] = landmark_ids
    df["Outlier"] = df["Outlier_Shape"] | df["Outlier_Landmark"]

    print(df["Outlier"].value_counts())
    return df