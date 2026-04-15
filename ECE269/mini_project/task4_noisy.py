import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
import os
from omp import orthogonal_matching_pursuit
from utils import generate_data, normalized_error

def single_trial_noisy(M, N, s, sigma):
    A, x, y, _ = generate_data(M, N, s, snr_sigma=sigma)
    
    A = np.random.randn(M, N)
    norm_A = np.linalg.norm(A, axis=0)
    norm_A[norm_A == 0] = 1 
    A = A / norm_A
    
    all_indices = np.arange(N)
    support = np.random.choice(all_indices, size=s, replace=False)
    x = np.zeros(N)
    magnitudes = np.random.uniform(1, 10, size=s)
    signs = np.random.choice([-1, 1], size=s)
    x[support] = magnitudes * signs
    
    n = np.random.normal(0, sigma, size=M)
    y = A @ x + n
    noise_norm = np.linalg.norm(n)
    
    x_hat_a, _, _ = orthogonal_matching_pursuit(A, y, 'sparsity', s)
    err_a = normalized_error(x, x_hat_a)
    success_a = 1 if err_a < 1e-3 else 0
    
    x_hat_b, _, _ = orthogonal_matching_pursuit(A, y, 'error', noise_norm)
    err_b = normalized_error(x, x_hat_b)
    success_b = 1 if err_b < 1e-3 else 0
    
    return success_a, success_b

def run_monte_carlo_noisy(M, N, s, sigma, num_trials):
    results = Parallel(n_jobs=-1)(delayed(single_trial_noisy)(M, N, s, sigma) for _ in range(num_trials))
    successes_a = [r[0] for r in results]
    successes_b = [r[1] for r in results]
    return np.mean(successes_a), np.mean(successes_b)

def plot_heatmap(data, N, sigma, case_name, filename):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, origin='lower', aspect='auto', cmap='viridis', extent=[1, N, 1, N], vmin=0, vmax=1)
    plt.colorbar(label='Probability of Success (Error < 1e-3)')
    plt.xlabel('Measurements (M)')
    plt.ylabel('Sparsity (s)')
    plt.title(f'Noisy Phase Transition (N={N}, $\\sigma$={sigma})\n{case_name}')
    plt.plot([0, N], [0, N/2], 'r--', linewidth=2)
    plt.savefig(f'plots/{filename}')
    plt.close()

def main():
    if not os.path.exists('plots'):
        os.makedirs('plots')

    Ns = [20, 50, 100]
    sigmas = [0.1, 1.0]
    num_trials = 2000

    for N in Ns:
        print(f"=== Processing N={N} ===")
        for sigma in sigmas:
            print(f"  sigma={sigma}...")
            start_time = time.time()
            
            grid_a = np.zeros((N, N))
            grid_b = np.zeros((N, N))
            
            for M in range(1, N + 1):
                if M % 10 == 0 or M == 1:
                    print(f"    M={M}/{N}")
                    
                for s in range(1, M + 1):
                    prob_a, prob_b = run_monte_carlo_noisy(M, N, s, sigma, num_trials)
                    grid_a[s-1, M-1] = prob_a
                    grid_b[s-1, M-1] = prob_b
            
            plot_heatmap(grid_a, N, sigma, 'Case A: Known Sparsity (s)', 
                         f'task4_N{N}_sig{sigma}_CaseA.png')
            plot_heatmap(grid_b, N, sigma, 'Case B: Known Noise Norm (||n||)', 
                         f'task4_N{N}_sig{sigma}_CaseB.png')
            
            print(f"  Finished sigma={sigma} in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
