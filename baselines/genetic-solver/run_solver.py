"""
Run Sholomon et al. 2013 Genetic Algorithm Solver on GAP datasets.

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

from genetic_solver import solve_puzzle


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


def run_solver_on_dataset(data_dir: str, output_path: str, 
                         pop_size: int = 100,
                         n_generations: int = 1000,
                         mutation_rate: float = 0.01,
                         max_samples: int = None, 
                         seed: int = 42):
    """
    Run genetic algorithm solver on all puzzles in dataset.
    
    Args:
        data_dir: Path to dataset directory (e.g., /path/to/GAP-3/test)
        output_path: Path to save predictions
        pop_size: Population size for genetic algorithm
        n_generations: Maximum number of generations
        mutation_rate: Mutation probability
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
        print(f"Genetic Algorithm Parameters:")
        print(f"  Population size: {pop_size}")
        print(f"  Max generations: {n_generations}")
        print(f"  Mutation rate: {mutation_rate}")
        
        # Initialize results
        predictions = np.zeros((num_samples, num_pieces), dtype=np.int64)
        fitness_scores = np.zeros(num_samples)
        generations_used = np.zeros(num_samples, dtype=np.int64)
        times = np.zeros(num_samples)
        
        # Process each puzzle
        for idx in tqdm(range(num_samples), desc="Solving puzzles"):
            start_time = time.time()
            
            # Load puzzle pieces
            pieces_lab = load_puzzle_from_hdf5(puzzles_dataset, idx)
            
            try:
                # Run genetic algorithm
                solution, fitness, stats = solve_puzzle(
                    pieces_lab, 
                    grid_size,
                    pop_size=pop_size,
                    n_generations=n_generations,
                    mutation_rate=mutation_rate,
                    seed=seed + idx  # Different seed for each puzzle
                )
                
                # Flatten solution to 1D permutation
                predictions[idx] = solution.flatten()
                fitness_scores[idx] = fitness
                generations_used[idx] = stats['n_generations']
                
            except Exception as e:
                print(f"\nError: Solver failed for puzzle {idx}: {e}")
                print("Using identity permutation as fallback.")
                predictions[idx] = np.arange(num_pieces)
                fitness_scores[idx] = -np.inf
                generations_used[idx] = 0
            
            times[idx] = time.time() - start_time
        
        # Save results
        print(f"\nSaving predictions to: {output_path}")
        np.save(output_path, predictions)
        
        # Save additional info
        info_path = output_path.replace('.npy', '_info.npz')
        np.savez(info_path, 
                fitness_scores=fitness_scores,
                generations_used=generations_used,
                times=times,
                pop_size=pop_size,
                n_generations=n_generations,
                mutation_rate=mutation_rate)
        
        print(f"\nResults:")
        print(f"  Mean fitness: {fitness_scores[fitness_scores > -np.inf].mean():.2f} "
              f"± {fitness_scores[fitness_scores > -np.inf].std():.2f}")
        print(f"  Mean generations: {generations_used[generations_used > 0].mean():.1f} "
              f"± {generations_used[generations_used > 0].std():.1f}")
        print(f"  Mean time per puzzle: {times.mean():.2f}s ± {times.std():.2f}s")
        print(f"  Total time: {times.sum():.2f}s ({times.sum()/60:.1f} minutes)")


def main():
    parser = argparse.ArgumentParser(description='Run Genetic Algorithm Solver')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory (e.g., /path/to/GAP-3/test)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for predictions (.npy)')
    parser.add_argument('--pop_size', type=int, default=100,
                       help='Population size (default: 100)')
    parser.add_argument('--n_generations', type=int, default=1000,
                       help='Maximum number of generations (default: 1000)')
    parser.add_argument('--mutation_rate', type=float, default=0.01,
                       help='Mutation rate (default: 0.01)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', 
                exist_ok=True)
    
    run_solver_on_dataset(
        args.data_dir,
        args.output,
        args.pop_size,
        args.n_generations,
        args.mutation_rate,
        args.max_samples,
        args.seed
    )


if __name__ == '__main__':
    main()
