# ECE 269 Mini Project #1: Sparse Recovery via OMP

**Author:** Haomin Qi  
**Date:** Winter 2026

## Overview
This project implements the Orthogonal Matching Pursuit (OMP) algorithm to solve sparse approximation problems. It includes simulations for phase transitions (noiseless and noisy) and applications in image inpainting and audio recovery.

## Directory Structure
- `omp.py`: The core implementation of the OMP algorithm.
- `utils.py`: Helper functions for data generation and error calculation.
- `task*.py`: Executable scripts for each task specified in the project requirements.
- `pr5/` & `pr6/`: Data files for Task 5 and Task 6.
- `plots/`: Generated phase transition plots (Task 3 & 4).
- `results_task5/`: Recovered images (Task 5).
- `results_task6/`: Recovered audio files (Task 6).

## Dependencies
Ensure you have the following Python packages installed:
\`\`\`bash
pip install -r requirements.txt
\`\`\`
(Requires: `numpy`, `scipy`, `matplotlib`, `joblib`, `pillow`, `soundfile`)

## How to Run the Code

### 1. Sanity Check
Verifies the correctness of the OMP implementation on a small scale.
\`\`\`bash
python3 task0_sanity_check.py
\`\`\`

### 2. Task 3: Noiseless Phase Transition
Generates phase transition plots for N=20, 50, 100.
**Output:** Heatmaps saved in `plots/`.
\`\`\`bash
python3 task3_noiseless.py
\`\`\`
*Note: This utilizes parallel processing (joblib) but may still take a few minutes for N=100.*

### 3. Task 4: Noisy Phase Transition
Evaluates OMP robustness under noise ($\sigma=0.1, 1.0$) with two stopping criteria.
**Output:** Heatmaps saved in `plots/`.
\`\`\`bash
python3 task4_noisy.py
\`\`\`
**⚠️ WARNING:** This simulation is computationally intensive. The N=100 case involves 2000 Monte Carlo trials per grid point and **may take several hours to complete**. The script supports "resume" functionality; it will skip generating plots that already exist in the `plots/` folder.

### 4. Task 5: Image Decoding
Recovers a hidden text message from compressed measurements.
**Output:** Comparison images saved in `results_task5/`.
\`\`\`bash
python3 task5_image.py
\`\`\`

### 5. Task 6: Audio Decoding
Recovers a hidden audio message from compressed measurements using a varying number of measurements $K$.
**Output:** `.wav` files saved in `results_task6/`.
\`\`\`bash
python3 task6_audio.py
\`\`\`
*Observation:* The message becomes intelligible around $K=1000$.

## Results Summary
All experimental results discussed in the report (figures and audio files) are pre-generated and included in their respective folders for convenience.
