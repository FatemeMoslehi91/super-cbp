import numpy as np
import pandas as pd
import random
import time
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr


# Import helper function for loading data
from helper_func import load_junifer_store

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
        population.append(chromosome)
    
    return population

def get_brain_bounds(storage_name="HCP_R1_LR_VoxExtr_TianHip.hdf5", data_path="E:/INM-7/SuperCBP/Data"):
    """
    Extract brain coordinate bounds from the HDF5 storage file.
    
    :param storage_name: Name of the HDF5 file
    :param data_path: Path to the data directory
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
    
    # Calculate bounds for each dimension
    brain_bounds = [
        [np.min(x_coords), np.max(x_coords)],  # x bounds
        [np.min(y_coords), np.max(y_coords)],  # y bounds
        [np.min(z_coords), np.max(z_coords)]   # z bounds
    ]
    
    print("Brain bounds extracted from voxel coordinates:")
    print(f"X range: [{brain_bounds[0][0]}, {brain_bounds[0][1]}]")
    print(f"Y range: [{brain_bounds[1][0]}, {brain_bounds[1][1]}]")
    print(f"Z range: [{brain_bounds[2][0]}, {brain_bounds[2][1]}]")
    
    return brain_bounds



def tournament_selection(population, fitness_scores, num_parents, tournsize=2):
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

def crossover(parent1, parent2, crossover_rate=0.8):
    """
    Perform single-point crossover between two parents.
    
    :param parent1: First parent chromosome.
    :param parent2: Second parent chromosome.
    :param crossover_rate: Probability of performing crossover.
    :return: Tuple containing two offspring chromosomes.
    """
    # Create copies of parents
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()
    
    # Determine whether to perform crossover
    if random.random() < crossover_rate:
        # Choose a random crossover point
        crossover_point = random.randint(1, len(parent1) - 1)
        
        # Perform crossover (properly handle numpy arrays)
        offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    
    return offspring1, offspring2



def mutation(chromosome, mutation_rate=0.1, mutation_strength=5.0, brain_bounds=None):
    """
    Mutate a chromosome by shifting cluster centers.
    :param chromosome: The chromosome to be mutated.
    :param mutation_rate: Probability of each gene being mutated.
    :param mutation_strength: Scale of the mutation if using Gaussian mutation.
    :param brain_bounds: Min and max coordinates (x, y, z) in brain space.
                      Format: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    :return: Mutated chromosome.
    
    """
    mutated = chromosome.copy()
    
    for i in range(0, len(chromosome), 3):
        if np.random.random() < mutation_rate:
            shifts = np.random.normal(0, mutation_strength, 3)
            mutated[i:i+3] += shifts
            
            if brain_bounds is not None:
                min_bound, max_bound = brain_bounds
                mutated[i:i+3] = np.clip(mutated[i:i+3], min_bound, max_bound)
    
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

def calculate_cluster_time_series(vox_data, cluster_labels, n_clusters=3):
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
        # Extract data for this subject
        subject_data = vox_data.xs(subject_id, level='subject')
        
        # Calculate average signal for each cluster
        cluster_signals = []
        for cluster_id in range(n_clusters):
            voxel_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_voxels = subject_data.iloc[:, voxel_indices]
            avg_signal = cluster_voxels.mean(axis=1)
            cluster_signals.append(avg_signal)
        
        # Stack to form (time_points, n_clusters)
        cluster_time_series[subject_id] = np.column_stack(cluster_signals)
    
    return cluster_time_series

def build_feature_matrix(cluster_time_series, roi_data):
    """
    Build feature matrix from cluster and ROI time series correlations.
    
    :param cluster_time_series: Dictionary with subject IDs as keys and cluster time series as values.
    :param roi_data: Array of ROI time series of shape (n_subjects, n_timepoints, n_rois).
    :return: Feature matrix of shape (n_subjects, n_clusters * n_rois).
    """
    # Sort subject IDs for consistency
    subject_ids = sorted(cluster_time_series.keys())
    n_subjects = len(subject_ids)
    
    # Get dimensions
    n_clusters = cluster_time_series[subject_ids[0]].shape[1]
    _, _, n_rois = roi_data.shape
    
    # Initialize feature matrix
    X = np.zeros((n_subjects, n_clusters * n_rois))
    
    for i, subject_id in enumerate(subject_ids):
        cluster_ts = cluster_time_series[subject_id]
        roi_ts = roi_data[i]
        
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
                except:
                    corr = 0.0
                
                X[i, feat_idx] = corr
                feat_idx += 1
    
    return X

def evaluate_fitness(chromosome, vox_data, voxel_coords, roi_data, labels, num_clusters=3):
    """
    Evaluate the fitness of a chromosome.
    
    :param chromosome: Chromosome to evaluate.
    :param vox_data: DataFrame of voxel time series.
    :param voxel_coords: Array of voxel coordinates.
    :param roi_data: Array of ROI time series.
    :param labels: Array of class labels (gender).
    :param num_clusters: Number of clusters.
    :return: Fitness score (AUC).
    """
    # Convert chromosome to cluster centers
    centers = decode_chromosome(chromosome, num_clusters)
    
    # Print the cluster centers
    print("  Cluster centers:")
    for i, center in enumerate(centers):
        print(f"    Cluster {i+1}: {center}")
    
    # Create KMeans model with these centers
    kmeans = KMeans(n_clusters=num_clusters, init=centers, n_init=1)
    cluster_labels = kmeans.fit_predict(voxel_coords)
    
    # Print the number of voxels in each cluster
    cluster_counts = np.bincount(cluster_labels)
    print("  Voxels per cluster:")
    for i, count in enumerate(cluster_counts):
        print(f"    Cluster {i+1}: {count} voxels")
    
    # Calculate cluster time series
    cluster_time_series = calculate_cluster_time_series(vox_data, cluster_labels, num_clusters)
    
    # Build feature matrix
    X = build_feature_matrix(cluster_time_series, roi_data)
    
    # Evaluate classification performance using cross-validation
    model = LogisticRegression(max_iter=1000, random_state=42)
    try:
        scores = cross_val_score(model, X, labels, cv=5, scoring='roc_auc')
        auc = np.mean(scores)
        print(f"  Cross-validation AUC scores: {scores}")
    except Exception as e:
        print(f"  Error in evaluation: {str(e)}")
        # If there's an error in evaluation, return a low fitness
        scores = np.zeros(5)
        auc = 0.0
    
    return auc

def load_data():
    """Load and prepare data for the genetic algorithm."""
    print("Loading hippocampal voxel data...")
    hipp_storage, hipp_markers = load_junifer_store(hipp_store_name, data_path)
    vox_data = hipp_storage.read_df(feature_name=hipp_markers[0])
    
    # Extract voxel coordinates
    voxel_coords = np.array([eval(col) for col in vox_data.columns])
    print(f"Loaded {len(voxel_coords)} voxels from {len(vox_data.index.get_level_values('subject').unique())} subjects")
    
    print("\nLoading ROI data...")
    roi_storage, roi_markers = load_junifer_store(roi_store_name, data_path)
    roi_df = roi_storage.read_df(feature_name=roi_markers[0])
    
    # Convert to numpy array with shape (n_subjects, n_timepoints, n_rois)
    roi_subjects = roi_df.index.get_level_values('subject').unique()
    time_points = len(roi_df.xs(roi_subjects[0], level='subject'))
    n_rois = roi_df.shape[1]
    
    roi_data = np.zeros((len(roi_subjects), time_points, n_rois))
    for i, sid in enumerate(roi_subjects):
        subj_data = roi_df.xs(sid, level='subject')
        roi_data[i] = subj_data.values
    
    print(f"Loaded ROI data: shape = {roi_data.shape}")
    
    print("\nLoading gender labels...")
    df = pd.read_csv(labels_path)
    n_subjects = len(roi_subjects)
    labels = (df['Gender'].values == 'F').astype(int)[:n_subjects]
    print(f"Loaded {len(labels)} gender labels")
    
    return vox_data, voxel_coords, roi_data, labels

def genetic_algorithm(
    pop_size,
    num_clusters,
    num_generations,
    crossover_rate,
    mutation_rate,
    mutation_strength,
    tournament_size
):
    
    """
    Run the genetic algorithm to optimize cluster centers.
    
    :param pop_size: Population size.
    :param num_clusters: Number of clusters.
    :param num_generations: Number of generations to run.
    :param crossover_rate: Probability of crossover.
    :param mutation_rate: Probability of mutation for each gene.
    :param mutation_strength: Scale of mutation.
    :param tournament_size: Size of tournament for selection.
    :return: Best chromosome and its fitness.
    """
    # Load data
    vox_data, voxel_coords, roi_data, labels = load_data()
    
    # Get brain bounds for initialization
    brain_bounds = get_brain_bounds(hipp_store_name, data_path)
    
    # Initialize population
    print("\nInitializing population...")
    population = initialize_population(pop_size, num_clusters, brain_bounds)
    
    # Evaluate initial population
    print("Evaluating initial population...")
    fitness_scores = []
    for i, chrom in enumerate(population):
        print("\n==================================================")
        print(f" Generation #1 | Chromosome #{i+1}")
        print("==================================================")
        fitness = evaluate_fitness(chrom, vox_data, voxel_coords, roi_data, labels, num_clusters)
        fitness_scores.append(fitness)
        print(f"  Chromosome {i+1} fitness: {fitness:.4f}")
    
    best_fitness = max(fitness_scores)
    best_idx = fitness_scores.index(best_fitness)
    best_chromosome = population[best_idx]
    
    print(f"\nInitial best fitness: {best_fitness:.4f} (Chromosome {best_idx+1})")
    
    # Main GA loop
    for generation in range(num_generations):
        print(f"\n==================================================")
        print(f" GENERATION {generation+1}/{num_generations}")
        print("==================================================")
        start_time = time.time()
        
        # Create new population through selection, crossover, and mutation
        new_population = []
        
        # Elitism: Keep the best individual
        new_population.append(best_chromosome)
        print(f"Elitism: Keeping best chromosome with fitness {best_fitness:.4f}")
        
        # Create the rest of the new population
        while len(new_population) < pop_size:
            # Select parents
            parents = tournament_selection(population, fitness_scores, 2, tournament_size)
            
            # Perform crossover
            if len(parents) >= 2:
                offspring1, offspring2 = crossover(parents[0], parents[1], crossover_rate)
                
                # Perform mutation
                offspring1 = mutation(offspring1, mutation_rate, mutation_strength)
                offspring2 = mutation(offspring2, mutation_rate, mutation_strength)
                
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
            print("\n==================================================")
            print(f" Generation #{generation+1} | Chromosome #{i+1}")
            print("==================================================")
            fitness = evaluate_fitness(chrom, vox_data, voxel_coords, roi_data, labels, num_clusters)
            fitness_scores.append(fitness)
            print(f"  Chromosome {i+1} fitness: {fitness:.4f}")
        
        # Update best individual
        current_best_fitness = max(fitness_scores)
        current_best_idx = fitness_scores.index(current_best_fitness)
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_chromosome = population[current_best_idx]
            print(f"\nNew best solution found! Chromosome {current_best_idx+1} with fitness {best_fitness:.4f}")
        
        # Print generation statistics
        generation_time = time.time() - start_time
        print(f"\nGeneration {generation+1} completed in {generation_time:.2f} seconds")
        print(f"Best fitness: {best_fitness:.4f}")
        print(f"Average fitness: {np.mean(fitness_scores):.4f}")
    
    print("\n==================================================")
    print(" GENETIC ALGORITHM COMPLETED")
    print("==================================================")
    print(f"Best fitness achieved: {best_fitness:.4f}")
    
    # Decode the best chromosome into cluster centers
    best_centers = decode_chromosome(best_chromosome, num_clusters)
    print("\nBest cluster centers:")
    for i, center in enumerate(best_centers):
        print(f"Cluster {i+1}: {center}")
    
    return best_chromosome, best_fitness

if __name__ == "__main__":
    # Run the genetic algorithm
    best_chromosome, best_fitness = genetic_algorithm(
        pop_size=50,
        num_clusters=3,
        num_generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1,
        mutation_strength=5.0,
        tournament_size=3
    )
    
    print("\nFinal Results:")
    print(f"Best fitness (AUC): {best_fitness:.4f}")
    
