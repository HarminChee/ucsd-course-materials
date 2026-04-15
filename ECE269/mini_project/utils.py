import numpy as np

def generate_data(M, N, s, snr_sigma=0):
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
    
    n = np.zeros(M)
    if snr_sigma > 0:
        n = np.random.normal(0, snr_sigma, size=M)
        
    y = A @ x + n
    
    return A, x, y, support

def normalized_error(x_true, x_hat):
    norm_true = np.linalg.norm(x_true)
    if norm_true == 0:
        return 0.0 if np.linalg.norm(x_hat) == 0 else np.inf
    
    return np.linalg.norm(x_true - x_hat) / norm_true

def is_support_correct(true_support, estimated_support):
    return set(true_support) == set(estimated_support)
