#!/usr/bin/env python3
import numpy as np
import pandas as pd
import random
import time
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
import warnings
# Import helper function for loading data
from helper_func import load_junifer_store

warnings.filterwarnings("ignore", category=RuntimeWarning)
# File paths
hipp_store_name = "HCP_R1_LR_VoxExtr_TianHip.hdf5"  # Hippocampus data
roi_store_name = "HCP_R1_LR_ROIExtr_Power.hdf5"     # ROI data
data_path = "E:/INM-7/SuperCBP/Data"
labels_path = "E:/INM-7/SuperCBP/Data/hcp_ya_395unrelated_info.csv"  # Gender labels



def initialize_population(pop_size, num_clusters, brain_bounds):
    """
    Generates an initial population of chromosomes.
    
    :param pop_size: Number of individuals in the population.
    :param num_clusters: Number of clusters (K).
    :param brain_bounds: Min and max coordinates (x, y, z) in brain space.
                       Format: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    :returns: List of population's chromosomes, each chromosome is a numpy array
             of shape (num_clusters * 3,) representing K cluster centers in 3D space.
    """
    population = []
    
    # Extract min and max bounds for each dimension
    min_bound = np.repeat([bound[0] for bound in brain_bounds], num_clusters)  # Repeat for each cluster
    max_bound = np.repeat([bound[1] for bound in brain_bounds], num_clusters)  # Repeat for each cluster
    
    for _ in range(pop_size):
        # Each chromosome represents num_clusters centers
        # Each center has 3 coordinates (X, Y, Z)
        chromosome = np.random.uniform(
            low=min_bound,
            high=max_bound,
            size=(num_clusters * 3)  # num_clusters * 3 (X,Y,Z)
        )
        
        # Ensure x-coordinates are negative for left hemisphere
        for i in range(num_clusters):
            x_idx = i * 3  # Index of x-coordinate for this cluster
            if chromosome[x_idx] > 0:
                chromosome[x_idx] = -abs(chromosome[x_idx])
        
        population.append(chromosome)
    
    return population

def get_brain_bounds(storage_name="HCP_R1_LR_VoxExtr_TianHip.hdf5", data_path="E:/INM-7/SuperCBP/Data", padding=20):
    """
    Extract brain coordinate bounds from the HDF5 storage file and add 20mm padding in all directions.
    
    :param storage_name: Name of the HDF5 file
    :param data_path: Path to the data directory
    :param padding: Padding in mm to adjust voxel location bounds
    :returns: List of [min, max] bounds for each dimension [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    """
    # Load the storage file
    storage, marker_names = load_junifer_store(storage_name, data_path)
    
    # Load data from the first feature to get voxel coordinates
    vox_sig = storage.read_df(feature_name=marker_names[0])
    
    # Get coordinates from column labels (which are stored as string tuples)
    coordinates = [eval(col) for col in vox_sig.columns]
    coordinates = np.array(coordinates)  # Convert to numpy array for easier operations
    
    # Extract x, y, z coordinates
    x_coords = coordinates[:, 0]
    y_coords = coordinates[:, 1]
    z_coords = coordinates[:, 2]
    
    # Calculate bounds for each dimension with padding    
    brain_bounds = [
        [np.min(x_coords), np.max(x_coords)],  # x bounds
        [np.min(y_coords), np.max(y_coords)],  # y bounds
        [np.min(z_coords), np.max(z_coords)]   # z bounds
    ]

    # apply padding
    brain_bounds[brain_bounds < 0] = brain_bounds[brain_bounds < 0] - padding
    brain_bounds[brain_bounds > 0] = brain_bounds[brain_bounds > 0] + padding
    
    print("Brain bounds extracted from voxel coordinates (with 20mm padding):")
    print(f"X range: [{brain_bounds[0][0]}, {brain_bounds[0][1]}]")
    print(f"Y range: [{brain_bounds[1][0]}, {brain_bounds[1][1]}]")
    print(f"Z range: [{brain_bounds[2][0]}, {brain_bounds[2][1]}]")
    
    return brain_bounds



def tournament_selection(population, fitness_scores, num_parents, tournsize):
    """
    Select parents using tournament selection.
    
    :param population: List of chromosomes in the current generation.
    :param fitness_scores: Corresponding fitness scores for each chromosome.
    :param num_parents: Number of parents to be selected.
    :param tournsize: Size of each tournament.
    :return: List of selected parent chromosomes.
    """
    selected_parents = []
    
    # Convert fitness scores to numpy array
    fitness_scores = np.array(fitness_scores)
    
    for _ in range(num_parents):
        # Randomly select tournsize individuals
        aspirants_idx = [random.randint(0, len(population) - 1) for _ in range(tournsize)]
        
        # Select the one with the highest fitness
        aspirants_fitness = [fitness_scores[i] for i in aspirants_idx]
        winner_idx = aspirants_idx[np.argmax(aspirants_fitness)]
        
        # Add the winner to the selected parents
        selected_parents.append(population[winner_idx])
    
    return selected_parents

def crossover(parent1, parent2, crossover_rate):
    """
    Perform single-point crossover between two parents.
    
    :param parent1: First parent chromosome.
    :param parent2: Second parent chromosome.
    :param crossover_rate: Probability of performing crossover.
    :return: Tuple containing two offspring chromosomes.
    """
    # Create copies of parents
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()
    
    # Determine whether to perform crossover
    if random.random() < crossover_rate:
        # Choose a random crossover point
        crossover_point = random.randint(1, len(parent1) - 1)
        
        # Perform crossover using np.concatenate
        offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    
    return offspring1, offspring2

def mutation(chromosome, mutation_rate, mutation_scale):
    """
    Perform mutation on a chromosome.
    
    :param chromosome: The chromosome to be mutated.
    :param mutation_rate: Probability of each gene being mutated.
    :param mutation_scale: Scale of the mutation if using Gaussian mutation.
    :return: Mutated chromosome.
    """
    # Create a copy of the chromosome
    mutated = chromosome.copy()
    
    # Apply mutation to each gene based on mutation_rate
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add Gaussian noise to the gene
            sigma = mutation_scale * abs(mutated[i]) if mutated[i] != 0 else mutation_scale
            mutated[i] += random.gauss(0, sigma)
    
    return mutated

def decode_chromosome(chromosome, num_clusters=3):
    """
    Decode a chromosome into cluster centers.
    
    :param chromosome: List of values representing cluster centers.
    :param num_clusters: Number of clusters.
    :return: Numpy array of shape (num_clusters, 3) representing cluster centers.
    """
    # Reshape the chromosome into cluster centers
    centers = np.array(chromosome).reshape(num_clusters, 3)
    return centers

def calculate_cluster_time_series(vox_data, cluster_labels, n_clusters=4):
    """
    Calculate average time series for each cluster for each subject.
    
    :param vox_data: DataFrame containing voxel time series.
    :param cluster_labels: Array of cluster assignments for each voxel.
    :param n_clusters: Number of clusters.
    :return: Dictionary with subject IDs as keys and time series arrays as values.
    """
    cluster_time_series = {}
    subjects = vox_data.index.get_level_values('subject').unique()
    
    for subject_id in subjects:
        try:
            # Extract data for this subject
            subject_data = vox_data.xs(subject_id, level='subject')
            
            # Pre-process subject data to handle NaN and inf values
            subject_data = subject_data.replace([np.inf, -np.inf], np.nan)
            subject_data = subject_data.fillna(method='ffill')  # Forward fill
            subject_data = subject_data.fillna(method='bfill')  # Backward fill
            subject_data = subject_data.fillna(0)  # Fill any remaining NaNs with 0
            
            # Calculate average signal for each cluster
            cluster_signals = []
            for cluster_id in range(n_clusters):
                voxel_indices = np.where(cluster_labels == cluster_id)[0]
                if len(voxel_indices) == 0:
                    print(f"  Warning: No voxels found in cluster {cluster_id+1} for subject {subject_id}")
                    # Create a zero array with the same length as other time series
                    zero_signal = np.zeros(subject_data.shape[0])
                    cluster_signals.append(zero_signal)
                    continue
                
                cluster_voxels = subject_data.iloc[:, voxel_indices]
                
                # Calculate mean while handling any remaining issues
                avg_signal = cluster_voxels.mean(axis=1, skipna=True)
                
                # Convert any remaining NaN or inf to 0
                avg_signal = np.nan_to_num(avg_signal, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Ensure the signal is finite
                if not np.all(np.isfinite(avg_signal)):
                    print(f"  Warning: Non-finite values in cluster {cluster_id+1} for subject {subject_id}")
                    avg_signal = np.zeros_like(avg_signal)
                
                cluster_signals.append(avg_signal)
            
            # Stack to form (time_points, n_clusters)
            cluster_time_series[subject_id] = np.column_stack(cluster_signals)
            
            # Final check for any remaining NaN or inf values
            if not np.all(np.isfinite(cluster_time_series[subject_id])):
                print(f"  Warning: Final time series contains non-finite values for subject {subject_id}")
                cluster_time_series[subject_id] = np.nan_to_num(cluster_time_series[subject_id], 
                                                              nan=0.0, posinf=0.0, neginf=0.0)
        
        except Exception as e:
            print(f"  Error processing subject {subject_id}: {str(e)}")
            # Create a zero array for this subject
            time_points = vox_data.xs(subjects[0], level='subject').shape[0]
            cluster_time_series[subject_id] = np.zeros((time_points, n_clusters))
    
    return cluster_time_series

def build_feature_matrix(cluster_time_series, roi_data, subjects=None):
    """
    Build feature matrix from cluster and ROI time series correlations.
    
    :param cluster_time_series: Dictionary with subject IDs as keys and cluster time series as values.
    :param roi_data: Array of ROI time series of shape (n_subjects, n_timepoints, n_rois).
    :param subjects: List of subjects in the same order as roi_data. If None, will use sorted keys from cluster_time_series.
    :return: Feature matrix of shape (n_subjects, n_clusters * n_rois).
    """
    # Get subject IDs for consistency
    if subjects is None:
        subject_ids = sorted(cluster_time_series.keys())
    else:
        subject_ids = subjects
        
    n_subjects = len(subject_ids)
    
    # Get dimensions
    n_clusters = next(iter(cluster_time_series.values())).shape[1]  # Get first item's cluster count
    _, _, n_rois = roi_data.shape
    
    print(f"Building feature matrix with {n_subjects} subjects, {n_clusters} clusters, {n_rois} ROIs")
    
    # Initialize feature matrix
    X = np.zeros((n_subjects, n_clusters * n_rois))
    
    # Create a mapping from subject ID to index in roi_data
    if subjects is not None:
        subject_to_idx = {sid: i for i, sid in enumerate(subjects)}
    else:
        subject_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
    
    for s_idx, subject_id in enumerate(subject_ids):
        if subject_id not in cluster_time_series:
            print(f"  Warning: Subject {subject_id} not found in cluster time series, skipping")
            continue
            
        cluster_ts = cluster_time_series[subject_id]
        roi_idx = subject_to_idx.get(subject_id)
        if roi_idx is None:
            print(f"  Warning: Subject {subject_id} not found in ROI data, skipping")
            continue
            
        roi_ts = roi_data[roi_idx]
        
        # Calculate correlations
        feat_idx = 0
        for c_idx in range(n_clusters):
            c_signal = cluster_ts[:, c_idx]
            for r_idx in range(n_rois):
                r_signal = roi_ts[:, r_idx]
                
                # Calculate correlation
                try:
                    corr, _ = pearsonr(c_signal, r_signal)
                    if np.isnan(corr):
                        corr = 0.0
                except Exception as e:
                    corr = 0.0
                    print(f"  Error calculating correlation for subject {subject_id}: {str(e)}")
                
                X[s_idx, feat_idx] = corr
                feat_idx += 1
    
    print(f"Feature matrix shape: {X.shape}")
    
    return X

def evaluate_fitness(chromosome, vox_data, voxel_coords, roi_data, labels, num_clusters=4, subjects=None, generation=None, chrom_idx=None):
    """
    Evaluate the fitness of a chromosome using K-means clustering.
    
    :param chromosome: Chromosome to evaluate.
    :param vox_data: DataFrame of voxel time series.
    :param voxel_coords: Array of voxel coordinates.
    :param roi_data: Array of ROI time series.
    :param labels: Array of class labels (gender).
    :param num_clusters: Number of clusters.
    :param subjects: List of subject IDs in the same order as roi_data.
    :param generation: Current generation number (for printing)
    :param chrom_idx: Chromosome index in the population (for printing)
    :return: Fitness score (AUC).
    """
    # Print header if generation and chromosome info provided
    if generation is not None and chrom_idx is not None:
        print(f"\n{'='*60}")
        print(f"Generation #{generation} | Chromosome #{chrom_idx}")
        print(f"{'='*60}")
    
    # Convert chromosome to cluster centers
    centers = decode_chromosome(chromosome, num_clusters)
    
    # Print the cluster centers
    print("  Cluster centers:")
    for i, center in enumerate(centers):
        print(f"    Cluster {i+1}: {center}")
    
    # Initialize K-means with the chromosome centers as initial centers
    kmeans = KMeans(
        n_clusters=num_clusters,
        init=centers,  # Use chromosome centers as initial centers
        n_init=1,      # Only one initialization since we're providing centers
        max_iter=100,  # Maximum iterations
        random_state=42
    )
    
    # Fit K-means to voxel coordinates
    cluster_labels = kmeans.fit_predict(voxel_coords)
    
    # Get final cluster centers
    final_centers = kmeans.cluster_centers_
    
    # Get cluster sizes
    cluster_sizes = np.bincount(cluster_labels, minlength=num_clusters)
    
    # Print the number of voxels in each cluster
    print("  Voxels per cluster:")
    for i, count in enumerate(cluster_sizes):
        print(f"    Cluster {i+1}: {count} voxels")
    
    # Calculate cluster time series
    cluster_time_series = calculate_cluster_time_series(vox_data, cluster_labels, num_clusters)
    
    # Build feature matrix
    X = build_feature_matrix(cluster_time_series, roi_data, subjects)
    
    # Verify that X has no NaN values
    if np.isnan(X).any():
        print("  Warning: Feature matrix contains NaN values. Replacing with 0.")
        X = np.nan_to_num(X)
    
    # Check for features with near-zero variance
    feature_vars = np.var(X, axis=0)
    low_var_indices = np.where(feature_vars < 1e-10)[0]
    if len(low_var_indices) > 0:
        print(f"  Removing {len(low_var_indices)} features with near-zero variance")
        X = np.delete(X, low_var_indices, axis=1)
    
    # Evaluate classification performance using cross-validation
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=10,
        random_state=42
    )
    
    try:
        scores = cross_val_score(model, X, labels, cv=5, scoring='roc_auc')
        auc = np.mean(scores)
        print(f"  Cross-validation AUC scores: {scores}")
        if chrom_idx is not None:
            print(f"  Chromosome {chrom_idx} fitness: {auc:.3f}")
    except Exception as e:
        print(f"  Error in evaluation: {str(e)}")
        scores = np.zeros(5)
        auc = 0.0
    
    return auc

def load_data(hemisphere='left'):
    """Load and prepare data for the genetic algorithm.
    
    :param hemisphere: 'left' or 'right' to specify which hemisphere to use
    :return: vox_data, voxel_coords, roi_data, labels, valid_subjects
    """
    print("Loading hippocampal voxel data...")
    hipp_storage, hipp_markers = load_junifer_store(hipp_store_name, data_path)
    vox_data = hipp_storage.read_df(feature_name=hipp_markers[0])
    
    # Extract voxel coordinates
    voxel_coords = np.array([eval(col) for col in vox_data.columns])
    
    # Filter voxels based on hemisphere
    if hemisphere == 'left':
        mask = voxel_coords[:, 0] < 0  # Left hemisphere has negative x coordinates
    else:
        mask = voxel_coords[:, 0] > 0  # Right hemisphere has positive x coordinates
    
    voxel_coords = voxel_coords[mask]
    vox_data = vox_data.iloc[:, mask]
    
    print(f"Loaded {len(voxel_coords)} {hemisphere} hemisphere voxels from {len(vox_data.index.get_level_values('subject').unique())} subjects")
    
    print("\nLoading ROI data...")
    roi_storage, roi_markers = load_junifer_store(roi_store_name, data_path)
    roi_df = roi_storage.read_df(feature_name=roi_markers[0])
    
    # Get unique subjects from both datasets
    hipp_subjects = set(vox_data.index.get_level_values('subject').unique())
    roi_subjects = roi_df.index.get_level_values('subject').unique()
    print(f"Hippocampal data: {len(hipp_subjects)} subjects")
    print(f"ROI data: {len(roi_subjects)} subjects")
    
    # Find common subjects to use for both datasets
    common_subjects = sorted(list(hipp_subjects.intersection(roi_subjects)))
    print(f"Common subjects: {len(common_subjects)}")
    
    # Filter vox_data to include only common subjects
    vox_data = vox_data.loc[vox_data.index.get_level_values('subject').isin(common_subjects)]
    
    # Convert to numpy array with shape (n_subjects, n_timepoints, n_rois) for common subjects
    time_points = len(roi_df.xs(common_subjects[0], level='subject'))
    n_rois = roi_df.shape[1]
    
    roi_data = np.zeros((len(common_subjects), time_points, n_rois))
    for i, sid in enumerate(common_subjects):
        subj_data = roi_df.xs(sid, level='subject')
        roi_data[i] = subj_data.values
    
    print(f"Loaded ROI data: shape = {roi_data.shape}")
    
    print("\nLoading gender labels...")
    try:
        import os
        if not os.path.exists(labels_path):
            print(f"ERROR: Gender labels file not found at path: {labels_path}")
            return None, None, None, None, None
            
        df = pd.read_csv(labels_path)
        
        if 'Subject' not in df.columns or 'Gender' not in df.columns:
            print("ERROR: Gender labels file does not have the required columns (Subject, Gender).")
            print(f"Available columns: {df.columns.tolist()}")
            return None, None, None, None, None
        
        # Create mapping from subject ID to gender
        subject_to_gender = {str(row['Subject']): 1 if row['Gender'] == 'F' else 0 
                             for _, row in df.iterrows()}
        
        # Get gender labels for common subjects
        labels = []
        valid_subjects = []
        
        for sid in common_subjects:
            if str(sid) in subject_to_gender:
                labels.append(subject_to_gender[str(sid)])
                valid_subjects.append(sid)
            
        print(f"Found gender labels for {len(valid_subjects)} out of {len(common_subjects)} common subjects")
        print(f"Gender distribution: {labels.count(1)}/{len(labels)} females ({labels.count(1)/len(labels)*100:.1f}%)")
        
        # Update vox_data and roi_data to include only subjects with gender labels
        vox_data = vox_data.loc[vox_data.index.get_level_values('subject').isin(valid_subjects)]
        
        # Rebuild roi_data for valid subjects only
        new_roi_data = np.zeros((len(valid_subjects), time_points, n_rois))
        for i, sid in enumerate(valid_subjects):
            subj_data = roi_df.xs(sid, level='subject')
            new_roi_data[i] = subj_data.values
            
        roi_data = new_roi_data
        
    except Exception as e:
        print(f"Error loading gender labels: {str(e)}")
        return None, None, None, None, None
    
    return vox_data, voxel_coords, roi_data, np.array(labels), valid_subjects

def genetic_algorithm(
    pop_size,
    num_clusters,
    num_generations,
    crossover_rate,
    mutation_rate,
    mutation_scale,
    tournament_size,
    hemisphere='left'  
):
    """
    Run the genetic algorithm to optimize cluster centers.
    
    :param pop_size: Population size.
    :param num_clusters: Number of clusters.
    :param num_generations: Number of generations to run.
    :param crossover_rate: Probability of crossover.
    :param mutation_rate: Probability of mutation for each gene.
    :param mutation_scale: Scale of mutation.
    :param tournament_size: Size of tournament for selection.
    :param hemisphere: 'left' or 'right' to specify which hemisphere to use.
    :return: Best chromosome and its fitness.
    """
    # Load data
    vox_data, voxel_coords, roi_data, labels, valid_subjects = load_data(hemisphere)
    
    # Check if data is loaded correctly
    if vox_data is None:
        print("Error loading data. Cannot continue.")
        return None, 0.0
    
    # Get brain bounds for initialization
    brain_bounds = get_brain_bounds(hipp_store_name, data_path)
    
    # Filter brain bounds based on hemisphere
    if hemisphere == 'left':
        brain_bounds[0][1] = 0  # Set max x to 0 for left hemisphere
    else:
        brain_bounds[0][0] = 0  # Set min x to 0 for right hemisphere
    
    # Initialize population
    print("\nInitializing population...")
    population = initialize_population(pop_size, num_clusters, brain_bounds)
    
    # Evaluate initial population
    print("Evaluating initial population...")
    fitness_scores = []
    for i, chrom in enumerate(population):
        fitness = evaluate_fitness(chrom, vox_data, voxel_coords, roi_data, labels, num_clusters, valid_subjects, 
                                 generation=1, chrom_idx=i+1)
        fitness_scores.append(fitness)
    
    best_fitness = max(fitness_scores)
    best_idx = fitness_scores.index(best_fitness)
    best_chromosome = population[best_idx]
    
    print(f"Initial best fitness: {best_fitness:.3f} (Chromosome {best_idx+1})")
    
    # Main GA loop
    for generation in range(num_generations):
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f" GENERATION {generation+1}/{num_generations}")
        print(f"{'='*60}")
        
        # Create new population through selection, crossover, and mutation
        new_population = []
        
        # Elitism: Keep the best individual
        new_population.append(best_chromosome)
        print(f"Elitism: Keeping best chromosome with fitness {best_fitness:.3f}")
        
        # Create the rest of the new population
        while len(new_population) < pop_size:
            # Select parents
            parents = tournament_selection(population, fitness_scores, 2, tournament_size)
            
            # Perform crossover
            if len(parents) >= 2:
                offspring1, offspring2 = crossover(parents[0], parents[1], crossover_rate)
                
                # Perform mutation
                offspring1 = mutation(offspring1, mutation_rate, mutation_scale)
                offspring2 = mutation(offspring2, mutation_rate, mutation_scale)
                
                # Add to new population
                new_population.append(offspring1)
                if len(new_population) < pop_size:
                    new_population.append(offspring2)
        
        # Replace old population
        population = new_population
        
        # Evaluate new population
        print(f"Evaluating population for generation {generation+1}...")
        fitness_scores = []
        for i, chrom in enumerate(population):
            fitness = evaluate_fitness(chrom, vox_data, voxel_coords, roi_data, labels, num_clusters, 
                                     valid_subjects, generation=generation+1, chrom_idx=i+1)
            fitness_scores.append(fitness)
        
        # Update best individual
        current_best_fitness = max(fitness_scores)
        current_best_idx = fitness_scores.index(current_best_fitness)
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_chromosome = population[current_best_idx]
            print(f"New best solution found! Chromosome {current_best_idx+1} with fitness {best_fitness:.3f}")
        
        # Print generation statistics
        generation_time = time.time() - start_time
        print(f"Generation {generation+1} completed in {generation_time:.2f} seconds")
        print(f"Best fitness: {best_fitness:.3f}")
        print(f"Average fitness: {np.mean(fitness_scores):.3f}")
    
    print("\nGenetic algorithm completed!")
    print(f"Best fitness achieved: {best_fitness:.3f}")
    
    # Decode the best chromosome into cluster centers
    best_centers = decode_chromosome(best_chromosome, num_clusters)
    print("Best cluster centers:")
    for i, center in enumerate(best_centers):
        print(f"Cluster {i+1}: {center}")
    
    # Save results to CSV file
    save_results(best_chromosome, best_fitness, num_clusters, valid_subjects, voxel_coords, vox_data, roi_data, labels)
    
    return best_chromosome, best_fitness

def plot_clusters_3d(voxel_coords, cluster_labels, centers, output_path="clustering_results/left_hemi_clusters_3d.html"):
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
            title="Left Hemisphere Hippocampus Clusters in 3D Space",
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

def save_results(chromosome, fitness, num_clusters, subjects, voxel_coords, vox_data, roi_data, labels):
    """
    Save the results of the genetic algorithm.
    
    :param chromosome: Best chromosome found.
    :param fitness: Fitness of the best chromosome.
    :param num_clusters: Number of clusters.
    :param subjects: List of subject IDs.
    :param voxel_coords: Voxel coordinates.
    :param vox_data: Voxel time series data.
    :param roi_data: ROI time series data.
    :param labels: Gender labels.
    """
    print("\nSaving results...")
    
    # Create directory if it doesn't exist
    import os
    from datetime import datetime
    
    # Get current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory with timestamp
    results_dir = f"clustering_results_{current_time}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Decode chromosome to get cluster centers
    centers = decode_chromosome(chromosome, num_clusters)
    
    # Use K-means to assign voxels to clusters
    kmeans = KMeans(
        n_clusters=num_clusters,
        init=centers,  # Use chromosome centers as initial centers
        n_init=1,      # Only one initialization since we're providing centers
        max_iter=100,  # Maximum iterations
        random_state=42
    )
    cluster_labels = kmeans.fit_predict(voxel_coords)
    
    # Calculate cluster time series
    cluster_time_series = calculate_cluster_time_series(vox_data, cluster_labels, num_clusters)
    
    # Build feature matrix
    X = build_feature_matrix(cluster_time_series, roi_data, subjects)
    
    # Save best chromosome
    np.save(f"{results_dir}/best_chromosome.npy", chromosome)
    
    # Save cluster assignments and coordinates
    cluster_data = pd.DataFrame({
        'x': voxel_coords[:, 0],
        'y': voxel_coords[:, 1],
        'z': voxel_coords[:, 2],
        'cluster': cluster_labels
    })
    cluster_data.to_csv(f"{results_dir}/cluster_assignments.csv", index=False)
    
    # Create and save 3D visualization
    plot_clusters_3d(voxel_coords, cluster_labels, centers, 
                    output_path=f"{results_dir}/clusters_3d.html")
    
    # Save a summary report
    with open(f"{results_dir}/summary.txt", "w") as f:
        f.write(f"Best fitness (AUC): {fitness:.3f}\n")
        f.write(f"Number of clusters: {num_clusters}\n")
        f.write(f"Number of subjects: {len(subjects)}\n")
        f.write(f"Gender distribution: {labels.sum()}/{len(labels)} females ({labels.sum()/len(labels)*100:.1f}%)\n\n")
        
        f.write("Best cluster centers:\n")
        for i, center in enumerate(centers):
            f.write(f"Cluster {i+1}: {center}\n")
        
        f.write(f"\nVoxels per cluster:\n")
        cluster_counts = np.bincount(cluster_labels)
        for i, count in enumerate(cluster_counts):
            f.write(f"Cluster {i+1}: {count} voxels\n")
            
    print(f"Results saved in directory: {results_dir}")
    print(f"Summary report saved to {results_dir}/summary.txt")
    print(f"Cluster assignments saved to {results_dir}/cluster_assignments.csv")
    print(f"Best chromosome saved to {results_dir}/best_chromosome.npy")
    print(f"3D visualization saved to {results_dir}/clusters_3d.html")

if __name__ == "__main__":
    # Run the genetic algorithm with updated parameters
    best_chromosome, best_fitness = genetic_algorithm(
        pop_size=2,               
        num_clusters=4,
        num_generations=2,        
        crossover_rate=0.8,
        mutation_rate=0.5,         
        mutation_scale=0.5,       
        tournament_size=3,
        hemisphere='left'
    )
    
    print("\nFinal Results:")
    print(f"Best fitness (AUC): {best_fitness:.3f}")
    print("Best chromosome saved to 'clustering_results/best_chromosome.npy'") 
