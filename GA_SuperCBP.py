import pandas as pd
import h5py
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import signal
from contextlib import contextmanager
import logging
import os
import platform
import threading
from threading import Timer
import time
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ga_algorithm.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Set environment variables
os.environ['OMP_NUM_THREADS'] = '2'

# File paths for cluster
CLUSTER_DATA_PATH = "/home/fmoslehi/data"  
CLUSTER_OUTPUT_PATH = "/home/fmoslehi/output"  

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    if platform.system() == 'Windows':
        # Windows implementation using Timer
        timer = Timer(seconds, lambda: threading.current_thread().interrupt_main())
        timer.start()
        try:
            yield
        except KeyboardInterrupt:
            raise TimeoutException("Timed out!")
        finally:
            timer.cancel()
    else:
        # Linux/Unix implementation using SIGALRM
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        
        # Register a signal handler
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

def load_partial_time_series(file_path, dataset_path):
    """
    Load time series from an HDF5 file.
    """
    time_series = []
    with h5py.File(file_path, "r") as f:
        keys = list(f[dataset_path].keys())
        for idx_name in keys:
            dataset_full_path = f"{dataset_path}/{idx_name}"
            if isinstance(f[dataset_full_path], h5py.Dataset):
                data = f[dataset_full_path][:]
                time_series.append(data)
    return np.array(time_series)

def compute_feature_space(cluster_centers, roi_time_series):
    """
    Compute feature space using correlations between cluster centers and ROI time series.
    """
    n_clusters = cluster_centers.shape[0]
    n_rois = roi_time_series.shape[2]
    n_subjects = roi_time_series.shape[0]
    features = np.zeros((n_clusters, n_rois, n_subjects))

    for i in range(n_clusters):
        for j in range(n_rois):
            for k in range(n_subjects):
                cluster_ts = np.mean(cluster_centers[i, :, :], axis=1)
                roi_ts = roi_time_series[k, :, j]
                correlation = np.corrcoef(cluster_ts, roi_ts)[0, 1]
                features[i, j, k] = correlation

    return features

def initialize_population(pop_size, num_clusters, brain_bounds, hip_time_series):
    """
    Generate initial population of chromosomes.
    Each chromosome represents cluster centers.
    """
    population = []
    min_bound, max_bound = brain_bounds
    
    # Calculate number of features dynamically from data shape
    n_features = hip_time_series.shape[1] * hip_time_series.shape[2]
    
    
    # Use current time as seed for random number generation
    np.random.seed(int(time.time() * 1000) % 10000)
    
    for _ in range(pop_size):
        # Each chromosome represents cluster centers with proper number of features
        chromosome = np.random.uniform(
            low=min_bound,
            high=max_bound,
            size=(num_clusters * n_features)  # Each cluster center has n_features dimensions
        )
        population.append(chromosome)
    
    return population

def tournament_selection(population, fitness_scores, num_parents, tournament_size=3):
    """
    Select parents using tournament selection.
    """
    selected_parents = []
    population_size = len(population)
    
    while len(selected_parents) < num_parents:
        tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        selected_parents.append(population[winner_idx])
    
    return selected_parents

def crossover(parent1, parent2):
    """
    Perform crossover between two parent chromosomes.
    """
    parent1, parent2 = np.array(parent1), np.array(parent2)
    if len(parent1) != len(parent2):
        raise ValueError("Parent chromosomes must have the same length")
    
    crossover_point = (len(parent1) // 2) // 3 * 3
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    
    return child1, child2

def mutate(chromosome, mutation_rate=0.1, mutation_strength=5.0, brain_bounds=None):
    """
    Mutate a chromosome by shifting cluster centers.
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

def evaluate_fitness(chromosome, roi_time_series, hip_time_series, n_clusters, run_number, generation_number, chromosome_number):
    """
    Evaluate clustering quality using classification AUC.
    """
    try:
        with timeout(300):  # 5 minutes timeout
            logging.info(f"\n{'='*50}")
            logging.info(f"Run #{run_number} | Generation #{generation_number} | Chromosome #{chromosome_number}")
            logging.info(f"{'='*50}")
            
            # Reshape chromosome into cluster centers
            n_features = hip_time_series.shape[1] * hip_time_series.shape[2]
            cluster_centers = chromosome.reshape(n_clusters, n_features)
            
            # Perform clustering using the chromosome's cluster centers
            logging.info("Performing clustering...")
            
            # Use a random state based on current time
            current_random_state = int(time.time() * 1000) % 10000
            
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters, 
                random_state=current_random_state, 
                batch_size=1024,
                init=cluster_centers,  # Use chromosome's centers as initial centers
                n_init=1 
            )
            
            # Reshape hip_time_series for MiniBatchKMeans
            n_subjects = hip_time_series.shape[0]
            hip_reshaped = hip_time_series.reshape(n_subjects, -1)
            
            # Fit  andpredict
            cluster_labels = kmeans.fit_predict(hip_reshaped)
            cluster_centers = kmeans.cluster_centers_.reshape(
                n_clusters, 
                hip_time_series.shape[1], 
                hip_time_series.shape[2]
            )
            logging.info(f"Cluster centers shape: {cluster_centers.shape}")
            
            # Compute feature space
            logging.info("Computing feature space...")
            features = compute_feature_space(cluster_centers, roi_time_series)
            logging.info(f"Feature space shape: {features.shape}")
            
            # Reshape features for classification
            n_clusters = features.shape[0]
            n_rois = features.shape[1]
            n_subjects = features.shape[2]
            X = features.transpose(2, 0, 1).reshape(n_subjects, n_clusters * n_rois)
            logging.info(f"Feature matrix shape: {X.shape}")
            
            # Load gender labels
            
            labels_path = "E:/INM-7/SuperCBP/Data/hcp_ya_395unrelated_info.csv"
            
           
            
            
            df = pd.read_csv(labels_path)
            y = (df['Gender'].values == 'F').astype(int)[:n_subjects]
            
            # Perform classification with AUC scoring
            clf = LogisticRegression(random_state=current_random_state, max_iter=1000)
            scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
            fitness_score = scores.mean()
            
            #logging.info(f"Gender distribution: {pd.Series(y).value_counts()}")
            #logging.info(f"Cross-validation AUC scores: {scores}")
            logging.info(f"Fitness score (mean AUC): {fitness_score:.3f}")
            logging.info(f"{'='*50}\n")
            
            return fitness_score
            
    except TimeoutException:
        logging.error(f"Fitness evaluation timed out for Run #{run_number} | Generation #{generation_number} | Chromosome #{chromosome_number}")
        return 0.0
    except Exception as e:
        logging.error(f"Error in fitness evaluation for Run #{run_number} | Generation #{generation_number} | Chromosome #{chromosome_number}: {str(e)}")
        return 0.0

def get_brain_bounds(file_path):
    """
    Extract absolute brain bounds from hippocampus HDF5 file.
    """
    with h5py.File(file_path, "r") as f:
        tianhip_path = "29f4c47760bd4864566a2095aa473237/key_data"
        keys = list(f[tianhip_path].keys())
        all_data = []
        
        for idx_name in keys:
            dataset_full_path = f"{tianhip_path}/{idx_name}"
            if isinstance(f[dataset_full_path], h5py.Dataset):
                data = f[dataset_full_path][:]
                all_data.append(data)
        
        all_data = np.array(all_data)
        absolute_min = np.min(all_data)
        absolute_max = np.max(all_data)
        
        print(f"Absolute minimum bound: {absolute_min}")
        print(f"Absolute maximum bound: {absolute_max}")
        return [absolute_min, absolute_max]

def genetic_algorithm(file_path_hip, file_path_roi, pop_size=50, num_clusters=3, 
                     num_generations=100, mutation_rate=0.1, mutation_strength=5.0, run_number=1):
    """
    Main genetic algorithm function.
    """
    logging.info(f"\n{'#'*70}")
    logging.info(f"Starting Run #{run_number}")
    logging.info(f"Population Size: {pop_size} | Number of Clusters: {num_clusters}")
    logging.info(f"Number of Generations: {num_generations}")
    logging.info(f"Mutation Rate: {mutation_rate} | Mutation Strength: {mutation_strength}")
    logging.info(f"{'#'*70}\n")
    
    # Set random seed based on current time
    np.random.seed(int(time.time() * 1000) % 10000)
    
    # Get brain bounds
    brain_bounds = get_brain_bounds(file_path_hip)
    
    # Load time series data
    tianhip_path = "29f4c47760bd4864566a2095aa473237/key_data"
    roi_path = "abb7161033609d7db16fe6efcddb123f/key_data"
    
    logging.info(f"Loading hip time series from: {file_path_hip}")
    hip_time_series = load_partial_time_series(file_path_hip, tianhip_path)
    logging.info(f"Hip time series shape: {hip_time_series.shape}")
    
    logging.info(f"Loading ROI time series from: {file_path_roi}")
    roi_time_series = load_partial_time_series(file_path_roi, roi_path)
    logging.info(f"ROI time series shape: {roi_time_series.shape}")
    
    # Initialize population
    population = initialize_population(pop_size, num_clusters, brain_bounds, hip_time_series)
    
    best_fitness = float('-inf')
    best_chromosome = None
    
    for generation in range(num_generations):
        logging.info(f"\n{'*'*50}")
        logging.info(f"Run #{run_number} | Generation {generation + 1}/{num_generations}")
        logging.info(f"{'*'*50}")
        
        # Evaluate fitness for each chromosome
        fitness_scores = []
        for chromosome_idx, chromosome in enumerate(population):
            fitness = evaluate_fitness(
                chromosome, 
                roi_time_series, 
                hip_time_series, 
                num_clusters,
                run_number,
                generation + 1,
                chromosome_idx + 1
            )
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_chromosome = chromosome.copy()
                logging.info(f"\nNew Best Fitness Found!")
                logging.info(f"Run #{run_number} | Generation {generation + 1} | Chromosome {chromosome_idx + 1}")
                logging.info(f"Best Fitness Score: {best_fitness:.3f}")
        
        logging.info(f"\nGeneration {generation + 1} Summary:")
        logging.info(f"Best fitness in generation: {best_fitness:.3f}")
        logging.info(f"Average fitness: {np.mean(fitness_scores):.3f}")
        logging.info(f"Min fitness: {np.min(fitness_scores):.3f}")
        logging.info(f"Max fitness: {np.max(fitness_scores):.3f}")
        
        # Selection
        num_parents = pop_size // 2
        selected_parents = tournament_selection(population, fitness_scores, num_parents)
        
        # Create new population
        new_population = []
        while len(new_population) < pop_size:
            # Select two different parents using indices
            parent_indices = np.random.choice(len(selected_parents), 2, replace=False)
            parent1 = selected_parents[parent_indices[0]]
            parent2 = selected_parents[parent_indices[1]]
            
            # Create children through crossover and mutation
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, mutation_strength, brain_bounds)
            child2 = mutate(child2, mutation_rate, mutation_strength, brain_bounds)
            new_population.extend([child1, child2])
        
        # Ensure we don't exceed population size
        population = new_population[:pop_size]
    
    logging.info(f"\n{'#'*70}")
    logging.info(f"Run #{run_number} Completed")
    logging.info(f"Final Best Fitness: {best_fitness:.3f}")
    logging.info(f"{'#'*70}\n")
    
    # Save results
    results_path = os.path.join(CLUSTER_OUTPUT_PATH, f"run_{run_number}_results.npz")
    np.savez(results_path, 
             best_chromosome=best_chromosome, 
             best_fitness=best_fitness)
    logging.info(f"Results saved to: {results_path}")
    
    return best_chromosome, best_fitness

if __name__ == "__main__":
    # File paths for cluster
    file_path_hip = "E:/INM-7/SuperCBP/Data/HCP_R1_LR_VoxExtr_TianHip.hdf5"
    file_path_roi = "E:/INM-7/SuperCBP/Data/HCP_R1_LR_ROIExtr_Power.hdf5"
    
    # Create output directory if it doesn't exist
    os.makedirs(CLUSTER_OUTPUT_PATH, exist_ok=True)
    
    # Run genetic algorithm
    best_chromosome, best_fitness = genetic_algorithm(
        file_path_hip=file_path_hip,
        file_path_roi=file_path_roi,
        pop_size=50,
        num_clusters=3,
        num_generations=100,
        mutation_rate=0.1,
        mutation_strength=5.0,
        run_number=1
    )
    
    logging.info("\nFinal Results:")
    logging.info(f"Best fitness score: {best_fitness:.3f}")
    logging.info(f"Best chromosome (cluster centers): {best_chromosome}")