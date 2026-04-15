import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
import os
from omp import orthogonal_matching_pursuit
from utils import generate_data, normalized_error, is_support_correct

def single_trial(M, N, s):
    A, x, y, true_support = generate_data(M, N, s, snr_sigma=0)
    x_hat, est_support, _ = orthogonal_matching_pursuit(A, y, 'sparsity', s)
    err = normalized_error(x, x_hat)
    success = is_support_correct(true_support, est_support)
    return success, err

def run_monte_carlo(M, N, s, num_trials):
    results = Parallel(n_jobs=-1)(delayed(single_trial)(M, N, s) for _ in range(num_trials))
    successes = [r[0] for r in results]
    errors = [r[1] for r in results]
    prob_recovery = np.mean(successes)
    avg_error = np.mean(errors)
    return prob_recovery, avg_error

def plot_heatmap(data, N, title, filename, cbar_label):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, origin='lower', aspect='auto', cmap='viridis', extent=[1, N, 1, N])
    plt.colorbar(label=cbar_label)
    plt.xlabel('Measurements (M)')
    plt.ylabel('Sparsity (s)')
    plt.title(f'{title} (N={N})')
    plt.plot([0, N], [0, N/2], 'r--', linewidth=2) 
    plt.savefig(f'plots/{filename}')
    plt.close()

def main():
    if not os.path.exists('plots'):
        os.makedirs('plots')

    Ns = [20, 50, 100]
    num_trials = 2000

    for N in Ns:
        print(f"Processing N={N}...")
        start_time = time.time()
        
        prob_recovery_grid = np.zeros((N, N))
        avg_error_grid = np.zeros((N, N))

        for M in range(1, N + 1):
            print(f"  N={N}, M={M}/{N}")
            for s in range(1, M + 1):
                prob, err = run_monte_carlo(M, N, s, num_trials)
                prob_recovery_grid[s-1, M-1] = prob
                avg_error_grid[s-1, M-1] = err
        
        plot_heatmap(prob_recovery_grid, N, 
                     'Probability of Exact Support Recovery', 
                     f'task3_N{N}_recovery.png', 
                     'Probability')
        
        plot_heatmap(avg_error_grid, N, 
                     'Average Normalized Error', 
                     f'task3_N{N}_error.png', 
                     'Error')

        print(f"Finished N={N} in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
