import numpy as np
import scipy.io
import scipy.io.wavfile
from PIL import Image
import os
from omp import orthogonal_matching_pursuit

Image.MAX_IMAGE_PIXELS = None

def load_data_task6(base_path):
    y_path = os.path.join(base_path, 'compressedSignal.mat')
    y_data = scipy.io.loadmat(y_path)
    y_key = [k for k in y_data.keys() if 'compressed' in k.lower()][0]
    y = y_data[y_key].astype(np.float64).flatten()

    A_path = os.path.join(base_path, 'compressionMatrix.mat')
    A_data = scipy.io.loadmat(A_path)
    A_key = [k for k in A_data.keys() if 'compression' in k.lower()][0]
    A = A_data[A_key].astype(np.float64)

    D_path = os.path.join(base_path, 'CompressedBasis.tiff')
    D_img = Image.open(D_path)
    D_arr = np.array(D_img).astype(np.float64)
    
    D = D_arr / 255.0 * 0.1284 - 0.0525
    
    return y, A, D

def solve_audio_recovery():
    if not os.path.exists('results_task6'):
        os.makedirs('results_task6')

    try:
        y, A, D = load_data_task6('pr6')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    fs = 7350
    sparsity_s = 100
    K_values = [10, 50, 100, 200, 300, 1000, 2000, 3000]
    
    scipy.io.wavfile.write('results_task6/original_compressed.wav', fs, y.astype(np.float32))

    Theta_full = A @ D

    for k in K_values:
        print(f"Processing K = {k}...")
        
        y_k = y[:k]
        Theta_k = Theta_full[:k, :]
        
        norms = np.linalg.norm(Theta_k, axis=0)
        norms[norms < 1e-10] = 1.0
        Theta_k_norm = Theta_k / norms
        
        s_hat, support, _ = orthogonal_matching_pursuit(Theta_k_norm, y_k, 'sparsity', sparsity_s)
        
        s_rec = s_hat / norms
        x_rec = D @ s_rec
        
        x_max = np.max(np.abs(x_rec))
        if x_max > 0:
            x_rec_norm = x_rec / x_max
        else:
            x_rec_norm = x_rec
            
        filename = f'results_task6/recovered_audio_K{k}.wav'
        scipy.io.wavfile.write(filename, fs, x_rec_norm.astype(np.float32))
        
        if k < 2000:
             s_ls, _, _, _ = np.linalg.lstsq(Theta_k, y_k, rcond=None)
             x_ls = D @ s_ls
             
             x_max_ls = np.max(np.abs(x_ls))
             if x_max_ls > 0: x_ls /= x_max_ls
             scipy.io.wavfile.write(f'results_task6/recovered_audio_K{k}_LS.wav', fs, x_ls.astype(np.float32))

if __name__ == "__main__":
    solve_audio_recovery()
