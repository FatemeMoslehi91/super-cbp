#!/usr/bin/env python3


import os
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy import stats
from statsmodels.stats.multitest import multipletests

from helper_func import load_junifer_store


store_name = "PIOP1_VoxExtr_TianS3_Hipp.hdf5"           # Hippocampus data 
roi_store_name = "PIOP1_ROIExtr_Power.hdf5"     # ROI data 
data_path = "/home/fmoslehi/SuperCBP/Hippocampus/"
labels_path = "/home/fmoslehi/SuperCBP/Hippocampus/piop1_labels.csv"  # Gender labels

# 4 clusters for the hippocampus 

# Define the specific markers we want to analyze Tian Parcellation
SELECTED_MARKERS = [
    "BOLD_VoxSigs_TianHipS3_head_m_lh_voxel_signal",
    "BOLD_VoxSigs_TianHipS3_head_l_lh_voxel_signal",
    "BOLD_VoxSigs_TianHipS3_body_lh_voxel_signal",
    "BOLD_VoxSigs_TianHipS3_tail_lh_voxel_signal"
]

"""
# Define the specific markers we want to analyze GA Parcellation
SELECTED_MARKERS = [
    "BOLD_VoxSigs_GA_Lh_Cluster_1_voxel_signal",
    "BOLD_VoxSigs_GA_Lh_Cluster_2_voxel_signal",
    "BOLD_VoxSigs_GA_Lh_Cluster_3_voxel_signal",
    "BOLD_VoxSigs_GA_Lh_Cluster_4_voxel_signal"
]
"""

OUTPUT_DIR = "/home/fmoslehi/SuperCBP/Univariate analysis/Hipp_PIOP1"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# --- Feature-building functions ---
# ---------------------------------------------------------------------
def compute_region_average_timeseries(storage, marker_names):
    print("\nComputing average time series for each region...")
    region_timeseries = {}
    vox_sig_first = storage.read_df(feature_name=marker_names[0])
    subjects = vox_sig_first.index.get_level_values('subject').unique()

    for subject in subjects:
        timepoints = len(vox_sig_first.xs(subject, level='subject'))
        region_averages = np.zeros((timepoints, len(marker_names)))
        for i, region_name in enumerate(marker_names):
            region_data = storage.read_df(feature_name=region_name)
            subject_data = region_data.xs(subject, level='subject')
            region_averages[:, i] = subject_data.mean(axis=1)
        region_timeseries[subject] = region_averages

    print(f"Computed average time series for {len(subjects)} subjects")
    return region_timeseries, subjects


def build_feature_matrix(region_timeseries, roi_df, subjects):
    print("\nBuilding feature matrix from region-ROI correlations...")
    n_subjects = len(subjects)
    first_subject = subjects[0]
    n_regions = region_timeseries[first_subject].shape[1]
    subject_roi = roi_df.xs(first_subject, level='subject')
    n_rois = subject_roi.shape[1]

    print(f"Number of subjects: {n_subjects}")
    print(f"Number of regions (clusters): {n_regions}")
    print(f"Number of ROIs: {n_rois}")
    print(f"Total features (region*ROI correlations): {n_regions * n_rois}")

    X = np.zeros((n_subjects, n_regions * n_rois))

    for s_idx, subject in enumerate(subjects):
        if s_idx % 20 == 0:
            print(f"Processing subject {s_idx+1}/{n_subjects} (ID: {subject})")

        region_ts = region_timeseries[subject]
        subject_roi = roi_df.xs(subject, level='subject')

        min_timepoints = min(len(region_ts), len(subject_roi))
        region_ts = region_ts[:min_timepoints]
        subject_roi = subject_roi.iloc[:min_timepoints]

        corr_matrix = np.zeros((n_regions, n_rois))
        for r_idx in range(n_regions):
            region_signal = region_ts[:, r_idx]
            for roi_idx in range(n_rois):
                roi_signal = subject_roi.iloc[:, roi_idx].values
                try:
                    corr, _ = pearsonr(region_signal, roi_signal)
                    corr_matrix[r_idx, roi_idx] = corr if not np.isnan(corr) else 0
                except Exception:
                    corr_matrix[r_idx, roi_idx] = 0

        np.save(f"{OUTPUT_DIR}/subject_{subject}_correlation.npy", corr_matrix)
        X[s_idx] = corr_matrix.flatten()

    print(f"Completed building feature matrix with shape {X.shape}")
    return X, n_regions, n_rois


# ---------------------------------------------------------------------
# --- Univariate group-difference analysis (t-test + FDR) ---
# ---------------------------------------------------------------------
def univariate_group_ttest(X, labels, feature_names, correction_method="fdr_bh", alpha=0.05):
    y = np.asarray(labels)
    groups = np.unique(y)
    if len(groups) != 2:
        raise ValueError(f"Expected exactly 2 groups, found {len(groups)}: {groups}")
    g1_mask = (y == groups[0])
    g2_mask = (y == groups[1])

    n_features = X.shape[1]
    t_stats = np.zeros(n_features)
    p_values = np.zeros(n_features)
    g1_means = np.zeros(n_features)
    g2_means = np.zeros(n_features)

    for i in range(n_features):
        x1 = X[g1_mask, i]
        x2 = X[g2_mask, i]
        t, p = stats.ttest_ind(x1, x2, equal_var=False)  # Welch's t-test
        t_stats[i] = t
        p_values[i] = p
        g1_means[i] = np.mean(x1)
        g2_means[i] = np.mean(x2)

    reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method=correction_method)

    results_df = pd.DataFrame({
        "feature": feature_names,
        f"mean_group{groups[0]}": g1_means,
        f"mean_group{groups[1]}": g2_means,
        "t_stat": t_stats,
        "p_value": p_values,
        "p_value_corrected": p_corrected,
        "significant": reject
    })
    return results_df.sort_values("p_value_corrected").reset_index(drop=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        start_time = time.time()

        print("Loading hippocampus (GA cluster) data...")
        hipp_storage, _ = load_junifer_store(store_name, data_path)

        print("\nLoading ROI data...")
        roi_storage, roi_markers = load_junifer_store(roi_store_name, data_path)
        roi_df = roi_storage.read_df(feature_name=roi_markers[0])

        region_timeseries, all_subjects = compute_region_average_timeseries(
            hipp_storage, SELECTED_MARKERS
        )

        hipp_subjects = set(all_subjects)
        roi_subjects = roi_df.index.get_level_values('subject').unique()
        common_subjects = sorted(list(hipp_subjects.intersection(set(roi_subjects))))
        print(f"Common subjects: {len(common_subjects)}")

        print("\nLoading gender labels...")
        labels_df = pd.read_csv(labels_path)
        subject_to_gender = {str(row['Subject']): 1 if row['Gender'] == 'F' else 0
                              for _, row in labels_df.iterrows()}

        labels = []
        valid_subjects = []
        for sid in common_subjects:
            if str(sid) in subject_to_gender:
                labels.append(subject_to_gender[str(sid)])
                valid_subjects.append(sid)
        labels = np.array(labels)
        print(f"Found gender labels for {len(valid_subjects)} out of {len(common_subjects)} subjects")
        print(f"Gender distribution: {labels.sum()}/{len(labels)} females "
              f"({labels.mean()*100:.1f}%)")

        # ---- Build (or rebuild) the feature matrix ----
        X, n_regions, n_rois = build_feature_matrix(region_timeseries, roi_df, valid_subjects)

        # Save immediately so you don't need to rebuild it again next time
        np.save(f"{OUTPUT_DIR}/feature_matrix.npy", X)
        pd.DataFrame({"Subject": valid_subjects, "Gender": labels}).to_csv(
            f"{OUTPUT_DIR}/labels.csv", index=False
        )
        print(f"\nSaved feature matrix to {OUTPUT_DIR}/feature_matrix.npy "
              f"(shape {X.shape}) and labels to {OUTPUT_DIR}/labels.csv")

        # ---- Univariate t-test + FDR correction ----
        feature_names = [
            f"Cluster{c+1}_ROI{r+1}"
            for c in range(n_regions)
            for r in range(n_rois)
        ]

        results_df = univariate_group_ttest(
            X, labels, feature_names,
            correction_method="fdr_bh",  # Benjamini-Hochberg FDR
            alpha=0.05
        )

        n_sig = results_df["significant"].sum()
        print(f"\n{n_sig} / {X.shape[1]} features significant after FDR correction (alpha=0.05)")
        print(results_df.head(15))

        results_df.to_csv(f"{OUTPUT_DIR}/univariate_results.csv", index=False)
        print(f"Saved full results table to {OUTPUT_DIR}/univariate_results.csv")

        elapsed_time = time.time() - start_time
        print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
