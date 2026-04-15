import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
from omp import orthogonal_matching_pursuit

def get_matrix_from_data(data, prefix, index):
    keys = list(data.keys())
    target = f"{prefix}{index}"
    
    if target in data:
        return data[target]
    
    target_lower = target.lower()
    if target_lower in data:
        return data[target_lower]
        
    for k in keys:
        if k.lower() == target_lower:
            return data[k]
            
    return None

def solve_image_recovery():
    if not os.path.exists('results_task5'):
        os.makedirs('results_task5')

    mat_path = os.path.join('pr5', 'Y1 Y2 Y3 and A1 A2 A3.mat')
    if not os.path.exists(mat_path):
        print(f"File not found: {mat_path}")
        return

    try:
        data = scipy.io.loadmat(mat_path)
    except Exception as e:
        print(f"Error loading .mat: {e}")
        return

    img_h, img_w = 90, 160
    cases = [1, 2, 3]
    
    for i in cases:
        print(f"Processing Case {i}...")
        
        y_raw = get_matrix_from_data(data, 'Y', i)
        A = get_matrix_from_data(data, 'A', i)
        
        if y_raw is None or A is None:
            print(f"Skipping Case {i}: Data not found in keys: {list(data.keys())}")
            continue

        y = y_raw.flatten()
        
        x_ls = np.linalg.pinv(A) @ y
        
        x_omp, support, _ = orthogonal_matching_pursuit(A, y, 'error', 1e-4)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        img_ls = x_ls.reshape((img_h, img_w), order='F')
        axes[0].imshow(img_ls, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title(f'Least Squares (Case {i})')
        
        img_omp = x_omp.reshape((img_h, img_w), order='F')
        axes[1].imshow(img_omp, cmap='gray')
        axes[1].axis('off')
        axes[1].set_title(f'OMP (Case {i}, s={len(support)})')
        
        plt.tight_layout()
        save_path = f'results_task5/case{i}_comparison.png'
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")

if __name__ == "__main__":
    solve_image_recovery()
