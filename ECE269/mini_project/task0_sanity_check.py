import numpy as np
import sys
import os

# Ensure current directory is in python path
sys.path.append(os.getcwd())

from omp import orthogonal_matching_pursuit
from utils import generate_data, normalized_error, is_support_correct

def run_test():
    print("=== Running OMP Sanity Check ===")
    N = 50
    M = 25
    s = 5
    sigma = 0 
    
    print(f"Params: N={N}, M={M}, s={s}, Noiseless")
    
    # Generate data
    np.random.seed(42) # Set seed for reproducibility
    A, x, y, true_support = generate_data(M, N, s, sigma)
    
    # Run OMP
    x_hat, est_support, _ = orthogonal_matching_pursuit(A, y, 'sparsity', s)
    
    # Check results
    error = normalized_error(x, x_hat)
    # Convert to set for comparison, ignore order
    support_match = is_support_correct(true_support, est_support)
    
    print(f"True Support:      {sorted(true_support)}")
    print(f"Estimated Support: {sorted(est_support)}")
    print(f"Normalized Error:  {error:.6e}")
    print(f"Exact Recovery?:   {support_match}")
    
    if error < 1e-5 and support_match:
        print("\n[SUCCESS] OMP implementation is correct!")
    else:
        print("\n[FAILURE] Something is wrong.")

if __name__ == "__main__":
    run_test()
