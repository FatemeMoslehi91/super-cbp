#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
import warnings
from helper_func import load_junifer_store
import plotly.graph_objects as go
import os
from datetime import datetime

warnings.filterwarnings("ignore", category=RuntimeWarning)

# File paths
hipp_store_name = "HCP_R1_LR_VoxExtr_TianHip.hdf5"  # Hippocampus data
roi_store_name = "HCP_R1_LR_ROIExtr_Power.hdf5"     # ROI data
data_path = "E:/INM-7/SuperCBP/Data"
labels_path = "E:/INM-7/SuperCBP/Data/hcp_ya_395unrelated_info.csv"  # Gender labels

# Given cluster centers
cluster_centers = np.array([
    
 [23.6466516  -16.10712299  -3.32568592],
 [ 5.84815204 -39.50161332 -15.02774733],
 [38.7346501  -30.08452042  -7.22486132],
 [19.21732393 -12.33279986  -0.07682786]

])

def filter_right_hemisphere(voxel_coords, vox_data):
    """
    Filter voxels to keep only those in the right hemisphere (x > 0).
    
    :param voxel_coords: Array of voxel coordinates
    :param vox_data: DataFrame containing voxel time series
    :return: Filtered voxel coordinates and voxel data
    """
    # Find indices of voxels in right hemisphere
    right_hemi_mask = voxel_coords[:, 0] > 0
    
    # Filter voxel coordinates
    filtered_coords = voxel_coords[right_hemi_mask]
    
    # Filter voxel data columns
    filtered_data = vox_data.iloc[:, right_hemi_mask]
    
    print(f"Filtered to {len(filtered_coords)} voxels in right hemisphere")
    
    return filtered_coords, filtered_data

def assign_voxels_to_clusters(voxel_coords, centers):
    """
    Assign voxels to the nearest cluster center.
    
    :param voxel_coords: Array of voxel coordinates
    :param centers: Array of cluster centers
    :return: Array of cluster assignments
    """
    n_clusters = len(centers)
    distances = np.zeros((len(voxel_coords), n_clusters))
    
    for i in range(n_clusters):
        diff = voxel_coords - centers[i]
        distances[:, i] = np.sqrt(np.sum(diff ** 2, axis=1))
    
    return np.argmin(distances, axis=1)

def calculate_cluster_time_series(vox_data, cluster_labels, n_clusters):
    """
    Calculate average time series for each cluster for each subject.
    
    :param vox_data: DataFrame containing voxel time series
    :param cluster_labels: Array of cluster assignments
    :param n_clusters: Number of clusters
    :return: Dictionary with subject IDs as keys and time series arrays as values
    """
    cluster_time_series = {}
    subjects = vox_data.index.get_level_values('subject').unique()
    
    for subject_id in subjects:
        subject_data = vox_data.xs(subject_id, level='subject')
        cluster_signals = []
        
        for cluster_id in range(n_clusters):
            voxel_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_voxels = subject_data.iloc[:, voxel_indices]
            avg_signal = cluster_voxels.mean(axis=1)
            cluster_signals.append(avg_signal)
        
        cluster_time_series[subject_id] = np.column_stack(cluster_signals)
    
    return cluster_time_series

def build_feature_matrix(cluster_time_series, roi_data, subjects=None):
    """
    Build feature matrix from cluster and ROI time series correlations.
    
    :param cluster_time_series: Dictionary with subject IDs as keys and cluster time series as values
    :param roi_data: Array of ROI time series
    :param subjects: List of subjects in the same order as roi_data
    :return: Feature matrix
    """
    if subjects is None:
        subject_ids = sorted(cluster_time_series.keys())
    else:
        subject_ids = subjects
        
    n_subjects = len(subject_ids)
    n_clusters = next(iter(cluster_time_series.values())).shape[1]
    _, _, n_rois = roi_data.shape
    
    X = np.zeros((n_subjects, n_clusters * n_rois))
    
    if subjects is not None:
        subject_to_idx = {sid: i for i, sid in enumerate(subjects)}
    else:
        subject_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
    
    for s_idx, subject_id in enumerate(subject_ids):
        if subject_id not in cluster_time_series:
            continue
            
        cluster_ts = cluster_time_series[subject_id]
        roi_idx = subject_to_idx.get(subject_id)
        if roi_idx is None:
            continue
            
        roi_ts = roi_data[roi_idx]
        
        feat_idx = 0
        for c_idx in range(n_clusters):
            c_signal = cluster_ts[:, c_idx]
            for r_idx in range(n_rois):
                r_signal = roi_ts[:, r_idx]
                
                try:
                    corr, _ = pearsonr(c_signal, r_signal)
                    if np.isnan(corr):
                        corr = 0.0
                except Exception:
                    corr = 0.0
                
                X[s_idx, feat_idx] = corr
                feat_idx += 1
    
    return X

def load_data():
    """Load and prepare data for evaluation."""
    print("Loading hippocampal voxel data...")
    hipp_storage, hipp_markers = load_junifer_store(hipp_store_name, data_path)
    vox_data = hipp_storage.read_df(feature_name=hipp_markers[0])
    
    voxel_coords = np.array([eval(col) for col in vox_data.columns])
    print(f"Loaded {len(voxel_coords)} voxels from {len(vox_data.index.get_level_values('subject').unique())} subjects")
    
    # Filter for right hemisphere
    voxel_coords, vox_data = filter_right_hemisphere(voxel_coords, vox_data)
    
    print("\nLoading ROI data...")
    roi_storage, roi_markers = load_junifer_store(roi_store_name, data_path)
    roi_df = roi_storage.read_df(feature_name=roi_markers[0])
    
    hipp_subjects = set(vox_data.index.get_level_values('subject').unique())
    roi_subjects = roi_df.index.get_level_values('subject').unique()
    print(f"Hippocampal data: {len(hipp_subjects)} subjects")
    print(f"ROI data: {len(roi_subjects)} subjects")
    
    common_subjects = sorted(list(hipp_subjects.intersection(roi_subjects)))
    print(f"Common subjects: {len(common_subjects)}")
    
    vox_data = vox_data.loc[vox_data.index.get_level_values('subject').isin(common_subjects)]
    
    time_points = len(roi_df.xs(common_subjects[0], level='subject'))
    n_rois = roi_df.shape[1]
    
    roi_data = np.zeros((len(common_subjects), time_points, n_rois))
    for i, sid in enumerate(common_subjects):
        subj_data = roi_df.xs(sid, level='subject')
        roi_data[i] = subj_data.values
    
    print(f"Loaded ROI data: shape = {roi_data.shape}")
    
    print("\nLoading gender labels...")
    try:
        if not os.path.exists(labels_path):
            print(f"ERROR: Gender labels file not found at path: {labels_path}")
            return None, None, None, None, None
            
        df = pd.read_csv(labels_path)
        
        if 'Subject' not in df.columns or 'Gender' not in df.columns:
            print("ERROR: Gender labels file does not have the required columns (Subject, Gender).")
            return None, None, None, None, None
        
        subject_to_gender = {str(row['Subject']): 1 if row['Gender'] == 'F' else 0 
                           for _, row in df.iterrows()}
        
        labels = []
        valid_subjects = []
        
        for sid in common_subjects:
            if str(sid) in subject_to_gender:
                labels.append(subject_to_gender[str(sid)])
                valid_subjects.append(sid)
            
        print(f"Found gender labels for {len(valid_subjects)} out of {len(common_subjects)} common subjects")
        print(f"Gender distribution: {labels.count(1)}/{len(labels)} females ({labels.count(1)/len(labels)*100:.1f}%)")
        
        vox_data = vox_data.loc[vox_data.index.get_level_values('subject').isin(valid_subjects)]
        
        new_roi_data = np.zeros((len(valid_subjects), time_points, n_rois))
        for i, sid in enumerate(valid_subjects):
            subj_data = roi_df.xs(sid, level='subject')
            new_roi_data[i] = subj_data.values
            
        roi_data = new_roi_data
        
    except Exception as e:
        print(f"Error loading gender labels: {str(e)}")
        return None, None, None, None, None
    
    return vox_data, voxel_coords, roi_data, np.array(labels), valid_subjects

def plot_clusters_3d(voxel_coords, cluster_labels, centers, output_path="clustering_results/right_hemi_clusters_3d.html"):
    """
    Create an interactive 3D plot of the clusters using plotly.
    
    :param voxel_coords: Array of voxel coordinates
    :param cluster_labels: Array of cluster assignments
    :param centers: Array of cluster centers
    :param output_path: Path to save the HTML plot
    """
    try:
        import plotly.graph_objects as go
        import os
        
        # Create figure
        fig = go.Figure()
        
        # Add voxels for each cluster with different colors
        colors = ['rgb(239,85,59)', 'rgb(99,110,250)', 'rgb(0,204,150)', 'rgb(255,192,0)']
        cluster_names = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
        
        for i in range(len(centers)):
            # Get voxels belonging to this cluster
            mask = cluster_labels == i
            cluster_voxels = voxel_coords[mask]
            
            # Add voxels as scatter points
            fig.add_trace(go.Scatter3d(
                x=cluster_voxels[:, 0],
                y=cluster_voxels[:, 1],
                z=cluster_voxels[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=colors[i],
                    opacity=0.7
                ),
                name=f"{cluster_names[i]} (n={np.sum(mask)})"
            ))
            
            # Add cluster center
            fig.add_trace(go.Scatter3d(
                x=[centers[i][0]],
                y=[centers[i][1]],
                z=[centers[i][2]],
                mode='markers',
                marker=dict(
                    size=10,
                    color=colors[i],
                    symbol='diamond',
                    opacity=1
                ),
                name=f"{cluster_names[i]} Center"
            ))
        
        # Update layout
        fig.update_layout(
            title="Right Hemisphere Hippocampus Clusters in 3D Space",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the plot
        fig.write_html(output_path)
        print(f"\nInteractive 3D plot saved to: {output_path}")
        
    except ImportError:
        print("\nError: plotly package is required for 3D visualization.")
        print("Please install it using: pip install plotly")
    except Exception as e:
        print(f"\nError creating 3D plot: {str(e)}")

def evaluate_clusters():
    """Evaluate clustering performance with given centers."""
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"clustering_results_right_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Open results file
    results_file = os.path.join(results_dir, "evaluation_results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Evaluation Results for Right Hemisphere Clustering\n")
        f.write("=" * 50 + "\n\n")
        
        # Save cluster centers
        f.write("Cluster Centers Used:\n")
        f.write("-" * 30 + "\n")
        for i, center in enumerate(cluster_centers):
            f.write(f"Cluster {i+1}: {center}\n")
        f.write("\n")
        
        # Load data
        f.write("Loading and Processing Data:\n")
        f.write("-" * 30 + "\n")
        vox_data, voxel_coords, roi_data, labels, valid_subjects = load_data()
        
        if vox_data is None:
            f.write("Error loading data. Cannot continue.\n")
            return
        
        # Assign voxels to clusters
        f.write("\nClustering Results:\n")
        f.write("-" * 30 + "\n")
        cluster_labels = assign_voxels_to_clusters(voxel_coords, cluster_centers)
        
        # Print cluster sizes
        cluster_counts = np.bincount(cluster_labels)
        f.write("\nVoxels per cluster:\n")
        for i, count in enumerate(cluster_counts):
            f.write(f"Cluster {i+1}: {count} voxels ({count/len(voxel_coords)*100:.1f}%)\n")
        
        # Calculate cluster time series
        f.write("\nCalculating cluster time series...\n")
        cluster_time_series = calculate_cluster_time_series(vox_data, cluster_labels, len(cluster_centers))
        
        # Build feature matrix
        f.write("\nBuilding feature matrix...\n")
        X = build_feature_matrix(cluster_time_series, roi_data, valid_subjects)
        
        # Evaluate classification performance
        f.write("\nClassification Performance:\n")
        f.write("-" * 30 + "\n")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=10,
            random_state=42
        )
        
        try:
            scores = cross_val_score(model, X, labels, cv=5, scoring='roc_auc')
            auc = np.mean(scores)
            std_auc = np.std(scores)
            f.write(f"\nCross-validation AUC scores:\n")
            for i, score in enumerate(scores):
                f.write(f"Fold {i+1}: {score:.3f}\n")
            f.write(f"\nAverage AUC: {auc:.3f} ± {std_auc:.3f}\n")
            
        except Exception as e:
            f.write(f"Error in evaluation: {str(e)}\n")
    
    # Create 3D visualization
    plot_path = os.path.join(results_dir, "clusters_3d.html")
    plot_clusters_3d(voxel_coords, cluster_labels, cluster_centers, output_path=plot_path)
    
    print(f"\nResults have been saved to: {results_file}")
    print(f"3D plot has been saved to: {plot_path}")
    print(f"All results are in directory: {results_dir}")

if __name__ == "__main__":
    evaluate_clusters() 