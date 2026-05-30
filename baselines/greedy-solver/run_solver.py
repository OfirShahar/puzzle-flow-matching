"""
Run Pomeranz et al. 2011 Greedy Solver on GAP datasets.

Usage:
    python run_solver.py --data_dir /path/to/GAP-3/test --output predictions.npy
"""

import os
import sys
import argparse
import numpy as np
import h5py
import time
from tqdm import tqdm
from skimage import color

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pomeranz_solver import solve_puzzle


def load_puzzle_from_hdf5(puzzles_dataset, idx: int):
    """
    Load a puzzle from HDF5 dataset.
    
    Args:
        puzzles_dataset: HDF5 dataset
        idx: Index of puzzle
    
    Returns:
        List of puzzle pieces in LAB color space
    """
    # Load puzzle (n_fragments, H, W, C) with values in [0, 255]
    puzzle = puzzles_dataset[idx]
    
    # Convert each piece to LAB color space
    pieces_lab = []
    for piece in puzzle:
        # Normalize to [0, 1] range
        piece_norm = piece / 255.0
        
        # Handle RGBA images by dropping alpha channel
        if piece_norm.shape[-1] == 4:
            piece_norm = piece_norm[..., :3]
        
        # Convert RGB to LAB
        piece_lab = color.rgb2lab(piece_norm)
        pieces_lab.append(piece_lab)
    
    return pieces_lab


def run_solver_on_dataset(data_dir: str, output_path: str, num_trials: int = 10, 
                         max_samples: int = None, seed: int = 42):
    """
    Run solver on all puzzles in dataset.
    
    Args:
        data_dir: Path to dataset directory (e.g., /path/to/GAP-3/test)
        output_path: Path to save predictions
        num_trials: Number of random seed trials per puzzle
        max_samples: Maximum number of samples to process (for testing)
        seed: Random seed
    """
    np.random.seed(seed)
    
    # Load HDF5 files
    puzzles_path = os.path.join(data_dir, 'puzzles.h5')
    labels_path = os.path.join(data_dir, 'labels_indices.h5')
    
    if not os.path.exists(puzzles_path):
        raise FileNotFoundError(f"Puzzles file not found: {puzzles_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    print(f"Loading data from: {data_dir}")
    
    with h5py.File(puzzles_path, 'r') as puzzles_file, \
         h5py.File(labels_path, 'r') as labels_file:
        
        puzzles_dataset = puzzles_file['puzzles']
        labels_dataset = labels_file['labels']
        
        num_samples = len(puzzles_dataset)
        if max_samples is not None:
            num_samples = min(num_samples, max_samples)
        
        # Get puzzle dimensions
        sample_puzzle = puzzles_dataset[0]
        num_pieces = sample_puzzle.shape[0]
        grid_size = int(np.sqrt(num_pieces))
        
        print(f"Dataset: {num_samples} puzzles")
        print(f"Puzzle size: {grid_size}x{grid_size} = {num_pieces} pieces")
        print(f"Piece size: {sample_puzzle.shape[1]}x{sample_puzzle.shape[2]}")
        print(f"Running {num_trials} trials per puzzle")
        
        # Initialize results
        predictions = np.zeros((num_samples, num_pieces), dtype=np.int64)
        scores = np.zeros(num_samples)
        times = np.zeros(num_samples)
        
        # Process each puzzle
        for idx in tqdm(range(num_samples), desc="Solving puzzles"):
            start_time = time.time()
            
            # Load puzzle pieces
            pieces_lab = load_puzzle_from_hdf5(puzzles_dataset, idx)
            
            # Try multiple random seeds and pick best solution
            best_solution = None
            best_score = -1
            
            # Generate random seeds for trials (limit to num_pieces if needed)
            actual_trials = min(num_trials, num_pieces)
            trial_seeds = np.random.choice(range(num_pieces), size=actual_trials, replace=False)
            
            for trial_seed in trial_seeds:
                try:
                    solution, score, iterations = solve_puzzle(
                        pieces_lab, grid_size, grid_size, seed=trial_seed
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_solution = solution
                
                except Exception as e:
                    print(f"\nWarning: Trial with seed {trial_seed} failed for puzzle {idx}: {e}")
                    continue
            
            if best_solution is None:
                print(f"\nError: All trials failed for puzzle {idx}. Using identity permutation.")
                best_solution = np.arange(num_pieces).reshape(grid_size, grid_size)
                best_score = 0.0
            
            # Flatten solution to 1D permutation
            predictions[idx] = best_solution.flatten()
            scores[idx] = best_score
            times[idx] = time.time() - start_time
        
        # Save results
        print(f"\nSaving predictions to: {output_path}")
        np.save(output_path, predictions)
        
        # Save additional info
        info_path = output_path.replace('.npy', '_info.npz')
        np.savez(info_path, 
                scores=scores,
                times=times,
                num_trials=num_trials)
        
        print(f"\nResults:")
        print(f"  Mean best buddies score: {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"  Mean time per puzzle: {times.mean():.2f}s ± {times.std():.2f}s")
        print(f"  Total time: {times.sum():.2f}s")


def main():
    parser = argparse.ArgumentParser(description='Run Pomeranz Greedy Solver')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory (e.g., /path/to/GAP-3/test)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for predictions (.npy)')
    parser.add_argument('--num_trials', type=int, default=10,
                       help='Number of random seed trials per puzzle')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    run_solver_on_dataset(
        args.data_dir,
        args.output,
        args.num_trials,
        args.max_samples,
        args.seed
    )


if __name__ == '__main__':
    main()
