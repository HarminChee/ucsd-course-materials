import numpy as np

def orthogonal_matching_pursuit(A, y, stopping_criterion, stopping_val, verbose=False):
    M, N = A.shape
    residual = y.copy()
    support = []
    x_hat = np.zeros(N)
    residuals_history = [np.linalg.norm(residual)]
    
    k = 0
    max_iter = M 
    x_S = np.array([])
    
    while k < max_iter:
        if stopping_criterion == 'sparsity':
            if len(support) >= stopping_val:
                break
        elif stopping_criterion == 'error':
            if np.linalg.norm(residual) <= stopping_val:
                break
        
        if k == 0 and len(support) == 0:
             correlations = A.T @ residual
        else:
             correlations = A.T @ residual
        
        best_atom_idx = np.argmax(np.abs(correlations))
        
        if np.abs(correlations[best_atom_idx]) < 1e-10:
            break

        if best_atom_idx in support:
            break
            
        support.append(best_atom_idx)
        
        A_S = A[:, support]
        x_S, _, _, _ = np.linalg.lstsq(A_S, y, rcond=None)
        
        fitted_y = A_S @ x_S
        residual = y - fitted_y
        
        residuals_history.append(np.linalg.norm(residual))
        k += 1
        
    if len(support) > 0:
        x_hat[support] = x_S
        
    return x_hat, support, residuals_history
