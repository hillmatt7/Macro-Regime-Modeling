# optimization_utils.py
"""
Optimization utilities for efficient computation in the macro regime pipeline
Includes parallel processing, memory management, and performance optimizations
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import warnings
import gc
import psutil
import os
from typing import Dict, Tuple, List
import time
from joblib import Parallel, delayed, dump, load, parallel_backend
import tempfile

warnings.filterwarnings('ignore')


def get_available_memory_gb():
    """Get available system memory in GB"""
    return psutil.virtual_memory().available / (1024**3)


def get_optimal_n_jobs():
    """Determine optimal number of parallel jobs based on system resources"""
    n_cpus = mp.cpu_count()
    available_memory = get_available_memory_gb()
    
    # Use at most n_cpus - 1 to keep system responsive
    max_jobs = max(1, n_cpus - 1)
    
    # Limit based on memory (assume each job needs ~1GB)
    memory_limited_jobs = max(1, int(available_memory / 1.5))
    
    return min(max_jobs, memory_limited_jobs)


def bootstrap_sample_worker(args: Tuple) -> float:
    """
    Worker function for parallel bootstrap sampling
    Returns ARI score for a single bootstrap sample
    """
    tensor, k_star, reference_labels, random_seed, sample_idx, params = args
    
    try:
        n_samples = len(tensor)
        
        # Set random seed for this worker
        np.random.seed(random_seed + sample_idx)
        
        # Resample with replacement
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_tensor = tensor[bootstrap_indices]
        
        # Fit GMM with optimized parameters
        gmm = GaussianMixture(
            n_components=k_star,
            covariance_type=params['covariance_type'],
            n_init=params['n_init'],  # Reduced for speed
            random_state=random_seed + sample_idx,
            max_iter=params['max_iter'],
            reg_covar=params['reg_covar'],
            tol=params['tol'],
            warm_start=False
        )
        
        gmm.fit(bootstrap_tensor)
        
        if not gmm.converged_:
            return 0.0
        
        # Predict on original data
        bootstrap_labels = gmm.predict(tensor)
        
        # Calculate ARI
        ari = adjusted_rand_score(reference_labels, bootstrap_labels)
        
        return ari
        
    except Exception as e:
        # Return 0 for failed samples
        return 0.0


def parallel_bootstrap_stability(tensor: np.ndarray, 
                               k_star: int,
                               reference_labels: np.ndarray,
                               n_samples: int = 500,
                               random_seed: int = 42,
                               covariance_type: str = 'full',
                               reg_covar: float = 1e-6) -> Dict:
    """
    Parallel implementation of bootstrap stability check
    """
    start_time = time.time()
    
    # Determine optimal parallelization

    target_mem_gb = 0.8 * psutil.virtual_memory().total / (1024**3)
    bytes_per_worker = tensor.nbytes * 0.05   # GMM uses ~5 % extra per copy
    max_workers_mem = int(target_mem_gb / (bytes_per_worker / (1024**3)))
    n_jobs = min(mp.cpu_count(), max_workers_mem)
    n_jobs = max(1, n_jobs)

    print(f"  Using {n_jobs} parallel workers for bootstrap validation")

    # For large bootstrap samples, use faster parameters
    gmm_params = {
        'covariance_type': covariance_type,
        'n_init': 10,  
        'max_iter': 500, 
        'reg_covar': reg_covar,
        'tol': 1e-3  # Increased tolerance
    }
    
    # Prepare arguments for parallel execution
    worker_args = [
        (tensor, k_star, reference_labels, random_seed, i, gmm_params)
        for i in range(n_samples)
    ]
    
    # ------------------------------------------------------------------
    # New high-throughput, low-overhead execution using joblib + memmap
    # ------------------------------------------------------------------
    aris = []

    # 1) Dump the tensor once to a temporary memmap file
    fd, mmap_path = tempfile.mkstemp(suffix=".mmap")
    os.close(fd)                           # we only need the filename
    dump(tensor.astype(np.float64), mmap_path)   # ensure double precision
    tensor_mmap = load(mmap_path, mmap_mode="r")

    # 2) Wrap the worker so joblib only passes an int index (cheap)
    def _boot(idx: int) -> float:
        return bootstrap_sample_worker(
            (tensor_mmap, k_star, reference_labels,
             random_seed, idx, gmm_params)
        )

    start = time.time()
    with parallel_backend("loky", inner_max_num_threads=1):
        aris = Parallel(
            n_jobs=n_jobs,
            prefer="processes",
            batch_size="auto",           # let joblib tune chunk size
            verbose=10                   # progress bar
        )(delayed(_boot)(i) for i in range(n_samples))
    elapsed_total = time.time() - start

    # 3) Clean up the memmap
    os.remove(mmap_path)
    
    # Clean up memory
    gc.collect()
    
    elapsed_total = time.time() - start_time
    print(f"  Bootstrap validation completed in {elapsed_total:.1f}s "
          f"({elapsed_total/n_samples:.2f}s per sample)")
    
    # Calculate statistics
    aris = np.array(aris)
    successful_aris = aris[aris > 0]
    
    return {
        'aris': aris.tolist(),
        'mean_ari': float(np.mean(aris)),
        'std_ari': float(np.std(aris)),
        'min_ari': float(np.min(aris)),
        'max_ari': float(np.max(aris)),
        'n_successful': len(successful_aris),
        'convergence_rate': len(successful_aris) / n_samples,
        'execution_time': elapsed_total,
        'n_workers': n_jobs
    }


def optimized_gmm_fit(tensor: np.ndarray, k: int, params: Dict) -> 'GMMCandidate':
    """
    Optimized GMM fitting with memory management
    """
    # Check available memory
    if get_available_memory_gb() < 1.0:
        gc.collect()  # Force garbage collection
        
    # Use mini-batch fitting for large datasets
    if len(tensor) > 10000:
        # Use mini-batch version
        from sklearn.mixture import GaussianMixture
        
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=params.get('covariance_type', 'full'),
            n_init=params.get('n_init', 10),
            random_state=params.get('random_seed', 42),
            max_iter=params.get('max_iter', 1000),
            reg_covar=params.get('reg_covar', 1e-6),
            warm_start=True  # Use warm start for mini-batches
        )
        
        # Fit in batches
        batch_size = 5000
        n_batches = (len(tensor) + batch_size - 1) // batch_size
        
        print(f"    Using mini-batch fitting with {n_batches} batches")
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tensor))
            batch = tensor[start_idx:end_idx]
            
            if i == 0:
                gmm.fit(batch)
            else:
                # Incremental fit
                gmm.fit(batch)
    else:
        # Standard fitting for smaller datasets
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=params.get('covariance_type', 'full'),
            n_init=params.get('n_init', 100),
            random_state=params.get('random_seed', 42),
            max_iter=params.get('max_iter', 1000),
            reg_covar=params.get('reg_covar', 1e-6)
        )
        
        gmm.fit(tensor)
    
    return gmm


def compute_icl_bic_aic(gmm: GaussianMixture, tensor: np.ndarray, k: int) -> Tuple[float, float, float]:
    """
    Efficiently compute ICL, BIC, and AIC for a fitted GMM
    """
    n_samples = len(tensor)
    n_features = tensor.shape[1]
    
    # Log-likelihood
    log_likelihood = gmm.score_samples(tensor).sum()
    
    # Count parameters
    if gmm.covariance_type == 'full':
        n_means = k * n_features
        n_cov = k * n_features * (n_features + 1) // 2
        n_weights = k - 1
        n_parameters = n_means + n_cov + n_weights
    else:
        # Simplified for other covariance types
        n_parameters = k * n_features * 2 + k - 1
    
    # BIC and AIC
    bic = -2 * log_likelihood + n_parameters * np.log(n_samples)
    aic = -2 * log_likelihood + 2 * n_parameters
    
    # ICL computation with memory-efficient approach
    # Get posterior probabilities in chunks to avoid memory issues
    chunk_size = min(1000, n_samples)
    entropy = 0.0
    
    for i in range(0, n_samples, chunk_size):
        chunk = tensor[i:i+chunk_size]
        posteriors = gmm.predict_proba(chunk)
        
        # Calculate entropy for this chunk
        log_posteriors = np.log(posteriors + 1e-10)
        chunk_entropy = -np.sum(posteriors * log_posteriors)
        entropy += chunk_entropy
    
    icl = bic - 2 * entropy
    
    return log_likelihood, bic, aic, icl


class MemoryMonitor:
    """Monitor memory usage and prevent crashes"""
    
    def __init__(self, threshold_gb: float = 1.0):
        self.threshold_gb = threshold_gb
        self.initial_memory = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    
    def check_memory(self, stage: str = ""):
        """Check if memory usage is within safe limits"""
        current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        available = get_available_memory_gb()
        
        if available < self.threshold_gb:
            print(f"  Warning: Low memory at {stage}: {available:.1f}GB available")
            gc.collect()  # Force garbage collection
            
            # Check again after GC
            available_after = get_available_memory_gb()
            if available_after < self.threshold_gb * 0.5:
                raise MemoryError(f"Insufficient memory: only {available_after:.1f}GB available")
        
        return available
    
    def log_usage(self, stage: str = ""):
        """Log current memory usage"""
        current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        delta = current_memory - self.initial_memory
        print(f"  Memory usage at {stage}: {current_memory:.1f}GB "
              f"(+{delta:.1f}GB from start)")


# Optimized version of elbow detection
def fast_elbow_detection(log_likelihoods: List[float], k_values: List[int]) -> int:
    """
    Fast elbow detection using second derivative method
    """
    if len(k_values) < 3:
        return k_values[0]
    
    # Calculate first differences
    first_diff = np.diff(log_likelihoods)
    
    # Calculate second differences
    second_diff = np.diff(first_diff)
    
    # Find the elbow as the point with maximum second derivative
    elbow_idx = np.argmax(second_diff) + 1  # +1 because of double diff
    
    # Ensure it's within bounds
    elbow_idx = max(1, min(elbow_idx, len(k_values) - 1))
    
    return k_values[elbow_idx]