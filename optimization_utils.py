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
from typing import Dict, Tuple, List, Optional
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
    
    # ICL computation
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

def single_gmm_worker_safe(args: Tuple) -> Tuple[int, Optional[Dict]]:
    """Memory-safe GMM worker for k-selection only"""
    k, tensor_path, params = args
    
    try:
        # Check memory before starting
        available_gb = psutil.virtual_memory().available / (1024**3)
        if available_gb < 0.5:
            return k, None
        
        # Load tensor from memmap (read-only)
        tensor = load(tensor_path, mmap_mode='r')
        
        # Fit GMM
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=params['covariance_type'],
            n_init=params['n_init'],
            random_state=params['random_seed'] + k,
            max_iter=params['max_iter'],
            init_params='kmeans',
            reg_covar=params['reg_covar'],
            warm_start=False,
            tol=params.get('tol', 1e-3)
        )
        
        gmm.fit(tensor)
        
        # Get predictions and probabilities
        labels = gmm.predict(tensor)
        posterior_probs = gmm.predict_proba(tensor)
        
        # Calculate metrics
        n_samples, n_features = tensor.shape
        log_likelihood = gmm.score_samples(tensor).sum()
        
        # Count parameters
        if params['covariance_type'] == 'full':
            n_parameters = k * n_features + k * n_features * (n_features + 1) // 2 + k - 1
        else:
            n_parameters = k * n_features * 2 + k - 1
        
        # BIC and AIC
        bic = -2 * log_likelihood + n_parameters * np.log(n_samples)
        aic = -2 * log_likelihood + 2 * n_parameters
        
        # ICL = BIC - 2 * entropy
        clipped_probs = np.clip(posterior_probs, 1e-10, 1.0)
        entropy = -np.sum(posterior_probs * np.log(clipped_probs))
        icl = bic - 2 * entropy
        
        # Return result with model parameters
        result = {
            'k': k,
            'log_likelihood': log_likelihood,
            'bic': bic,
            'icl': icl,
            'aic': aic,
            'labels': labels,
            'posterior_probs': posterior_probs,
            'converged': gmm.converged_,
            'n_iter': gmm.n_iter_,
            'means': gmm.means_,
            'covariances': gmm.covariances_,
            'weights': gmm.weights_,
            'precisions_cholesky': gmm.precisions_cholesky_
        }
        
        return k, result
        
    except Exception as e:
        print(f"GMM k={k} failed: {e}")
        return k, None


def parallel_gmm_grid_search_safe(tensor: np.ndarray,
                                k_min: int,
                                k_max: int,
                                n_init: int = 10,
                                max_iter: int = 1000,
                                covariance_type: str = 'full',
                                reg_covar: float = 1e-6,
                                random_seed: int = 42,
                                window_end: str = None) -> Dict:
    """Memory-safe parallel GMM k-selection"""
    from core import GMMCandidate
    import tempfile
    import os
    from joblib import Parallel, delayed, dump, load, parallel_backend
    
    start_time = time.time()
    
    # Calculate safe number of workers
    tensor_size_gb = tensor.nbytes / (1024**3)
    available_gb = psutil.virtual_memory().available / (1024**3)
    
    # Conservative memory estimation
    memory_per_worker = tensor_size_gb * 1.5 + 0.3  # Tensor + overhead
    max_workers_memory = max(1, int(available_gb * 0.6 / memory_per_worker))
    max_workers_cpu = max(1, psutil.cpu_count() - 1)
    n_workers = min(max_workers_memory, max_workers_cpu, k_max - k_min + 1)
    
    # Reduce n_init for large tensors
    if tensor_size_gb > 1.0:
        safe_n_init = max(5, n_init // 2)
    else:
        safe_n_init = n_init
    
    print(f"  Using {n_workers} workers with {safe_n_init} initializations each")
    print(f"  Tensor: {tensor_size_gb:.2f} GB, Available: {available_gb:.2f} GB")
    
    # Create temporary memmap
    fd, tensor_path = tempfile.mkstemp(suffix=".mmap")
    os.close(fd)
    dump(tensor.astype(np.float64), tensor_path)
    
    try:
        # Prepare parameters
        params = {
            'covariance_type': covariance_type,
            'n_init': safe_n_init,
            'max_iter': max_iter,
            'reg_covar': reg_covar,
            'random_seed': random_seed,
            'tol': 1e-3
        }
        
        # Prepare work items
        k_values = list(range(k_min, k_max + 1))
        work_items = [(k, tensor_path, params) for k in k_values]
        
        # Execute in parallel
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(
                n_jobs=n_workers,
                prefer="processes",
                batch_size=1,
                verbose=10
            )(delayed(single_gmm_worker_safe)(item) for item in work_items)
        
        # Build candidates dictionary
        candidates = {}
        for k, result in results:
            if result is not None:
                # Reconstruct GMM model
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=covariance_type,
                    random_state=random_seed
                )
                
                # Set fitted parameters
                gmm.weights_ = result['weights']
                gmm.means_ = result['means']
                gmm.covariances_ = result['covariances']
                gmm.precisions_cholesky_ = result['precisions_cholesky']
                gmm.converged_ = result['converged']
                gmm.n_iter_ = result['n_iter']
                gmm.lower_bound_ = result['log_likelihood']
                
                # Create candidate
                candidate = GMMCandidate(
                    k=k,
                    log_likelihood=result['log_likelihood'],
                    bic=result['bic'],
                    icl=result['icl'],
                    aic=result['aic'],
                    labels=result['labels'],
                    posterior_probs=result['posterior_probs'],
                    model=gmm,
                    converged=result['converged'],
                    n_iter=result['n_iter']
                )
                
                candidates[k] = candidate
        
        elapsed_time = time.time() - start_time
        print(f"  Parallel k-selection completed in {elapsed_time:.1f}s")
        print(f"  Successfully fitted {len(candidates)}/{len(k_values)} models")
        
        return candidates
        
    finally:
        # Clean up
        if os.path.exists(tensor_path):
            os.remove(tensor_path)
        gc.collect()