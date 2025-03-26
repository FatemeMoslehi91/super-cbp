import h5py
import numpy as np
from tslearn.clustering import TimeSeriesKMeans

# Function to load partial time series from an HDF5 file
def load_partial_time_series(file_path, dataset_path, max_samples=2):
    """
    Load a limited number of time series from an HDF5 file.

    Parameters:
        file_path (str): Path to the HDF5 file.
        dataset_path (str): Path to the dataset inside the HDF5 file.
        max_samples (int): Maximum number of samples to load.

    Returns:
        np.ndarray: Loaded time series as a NumPy array.
    """
    time_series = []
    with h5py.File(file_path, "r") as f:
        keys = list(f[dataset_path].keys())[:max_samples]  # Load a limited number of keys
        for idx_name in keys:
            dataset_full_path = f"{dataset_path}/{idx_name}"
            if isinstance(f[dataset_full_path], h5py.Dataset):
                data = f[dataset_full_path][:]  # Load the dataset
                time_series.append(data)
    return np.array(time_series)

# Function to compute the feature space using correlations
def compute_feature_space(cluster_centers, roi_time_series):
    """
    Compute the feature space using correlations between cluster centers and ROI time series.

    Parameters:
        cluster_centers (np.ndarray): Cluster centers from TimeSeriesKMeans (shape: [n_clusters, time_points, features]).
        roi_time_series (np.ndarray): ROI time series (shape: [n_subjects, time_points, n_rois]).

    Returns:
        np.ndarray: Feature space (shape: [n_clusters, n_rois, n_subjects]).
    """
    n_clusters = cluster_centers.shape[0]
    n_rois = roi_time_series.shape[2]
    n_subjects = roi_time_series.shape[0]

    # Initialize the feature space matrix
    features = np.zeros((n_clusters, n_rois, n_subjects))

    # Compute correlations
    for i in range(n_clusters):  # For each cluster
        for j in range(n_rois):  # For each ROI
            for k in range(n_subjects):  # For each subject
                cluster_ts = np.mean(cluster_centers[i, :, :], axis=1)  # Average across features
                roi_ts = roi_time_series[k, :, j]  # ROI time series for subject k
                correlation = np.corrcoef(cluster_ts, roi_ts)[0, 1]  # Compute correlation
                features[i, j, k] = correlation

    return features

# File paths
file_path_hip = "E:/INM-7/SuperCBP/Data/HCP_R1_LR_VoxExtr_TianHip.hdf5"
file_path_roi = "E:/INM-7/SuperCBP/Data/HCP_R1_LR_ROIExtr_Power.hdf5"

# Dataset paths
tianhip_path = "29f4c47760bd4864566a2095aa473237/key_data"
roi_path = "abb7161033609d7db16fe6efcddb123f/key_data"

# Load partial ROI time series
print("Loading partial ROI time series...")
roi_time_series = load_partial_time_series(file_path_roi, roi_path, max_samples=2)
print(f"Partial ROI time series shape: {roi_time_series.shape}")

# Load partial cluster time series from TianHip
print("Loading partial cluster time series from TianHip...")
cluster_time_series = load_partial_time_series(file_path_hip, tianhip_path, max_samples=2)
print(f"Partial cluster time series shape: {cluster_time_series.shape}")

# Perform clustering on the cluster time series
print("Clustering partial cluster time series...")
n_clusters = 3  # Number of clusters
ts_kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42, n_init=2)
cluster_labels = ts_kmeans.fit_predict(cluster_time_series)
cluster_centers = ts_kmeans.cluster_centers_
print(f"Cluster centers shape: {cluster_centers.shape}")

# Compute the feature space
print("Computing feature space (correlations)...")
features = compute_feature_space(cluster_centers, roi_time_series)

# Print the feature space
print(f"Feature space shape: {features.shape}")
print("Feature space (correlations):")
print(features)