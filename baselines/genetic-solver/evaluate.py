"""
Evaluate jigsaw puzzle solver predictions.

Usage:
    python evaluate.py --predictions predictions.npy --data_dir /path/to/GAP-3/test
"""

import os
import sys
import argparse
import numpy as np
import h5py
import json
from pathlib import Path
from typing import Dict

# Add parent directory to path for importing evaluate_predictions
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate_predictions import evaluate_predictions


def compute_metrics(predicted_perms: np.ndarray, ground_truth_perms: np.ndarray) -> Dict:
    """
    Compute comprehensive evaluation metrics using the unified evaluation framework.
    
    IMPORTANT: Handles permutation format conversion!
    - predicted_perms[b][pos] = piece_id (position → piece mapping from solver)
    - ground_truth_perms[b][piece] = pos (piece → position mapping from labels)
    
    These are INVERSE permutations! We convert predictions to piece→position format
    to match ground truth before comparison.
    
    Args:
        predicted_perms: (N, num_pieces) predicted arrangements (position → piece)
        ground_truth_perms: (N, num_pieces) ground truth labels (piece → position)
    
    Returns:
        dict with comprehensive metrics including spatial relationships
    """
    num_samples = predicted_perms.shape[0]
    num_pieces = predicted_perms.shape[1]
    grid_size = int(np.sqrt(num_pieces))
    
    # Convert predicted arrangements (position→piece) to piece→position format
    # to match ground truth labels format
    predicted_piece_to_pos = np.zeros_like(predicted_perms)
    for b in range(num_samples):
        for pos, piece_id in enumerate(predicted_perms[b]):
            predicted_piece_to_pos[b, piece_id] = pos
    
    # Now both are in piece→position format and can be compared
    # Use the comprehensive evaluation function
    results = evaluate_predictions(predicted_piece_to_pos, ground_truth_perms, grid_size)
    
    # Add per-sample statistics for backward compatibility
    correct_pieces = (predicted_piece_to_pos == ground_truth_perms).astype(float)
    per_sample_piece_acc = correct_pieces.mean(axis=1)
    
    # Extract perfect matches per sample
    perfect = (predicted_piece_to_pos == ground_truth_perms).all(axis=1)
    
    # Add legacy fields for backward compatibility
    results['legacy_format'] = {
        'perfect_rate': results['metrics']['perfect_accuracy'] / 100.0,
        'piece_accuracy': results['metrics']['absolute_accuracy'] / 100.0,
        'neighbor_accuracy': results['metrics']['spatial_relationship_accuracy'] / 100.0,
        'per_sample_piece_acc': per_sample_piece_acc.tolist(),
        'per_sample_perfect': perfect.tolist(),
    }
    
    return results


def load_ground_truth(data_dir: str) -> np.ndarray:
    """
    Load ground truth labels from HDF5.
    
    Args:
        data_dir: Path to dataset directory
    
    Returns:
        Ground truth permutations (N, num_pieces)
    """
    labels_path = os.path.join(data_dir, 'labels_indices.h5')
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    with h5py.File(labels_path, 'r') as f:
        labels = f['labels'][:]
    
    return labels


def main():
    parser = argparse.ArgumentParser(description='Evaluate puzzle solver predictions')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions file (.npy)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory containing ground truth')
    parser.add_argument('--output', type=str, default=None,
                       help='Optional: Path to save detailed metrics (.npz)')
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from: {args.predictions}")
    predictions = np.load(args.predictions)
    
    # Load ground truth
    print(f"Loading ground truth from: {args.data_dir}")
    ground_truth = load_ground_truth(args.data_dir)
    
    # Verify shapes match
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs ground truth {ground_truth.shape}"
        )
    
    print(f"Evaluating {predictions.shape[0]} puzzles with {predictions.shape[1]} pieces each")
    
    # Compute metrics
    print("\nComputing comprehensive evaluation metrics...")
    results = compute_metrics(predictions, ground_truth)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nDataset: {results['dataset_info']['num_puzzles']} puzzles, "
          f"{results['dataset_info']['num_pieces_per_puzzle']} pieces each "
          f"({results['dataset_info']['grid_size']}x{results['dataset_info']['grid_size']} grid)")
    print("\n" + "-" * 60)
    print("METRICS")
    print("-" * 60)
    print(f"Perfect Accuracy:              {results['metrics']['perfect_accuracy']:6.2f}% "
          f"({results['detailed_stats']['perfect_puzzles']}/{results['detailed_stats']['total_puzzles']} puzzles)")
    print(f"Absolute Accuracy:             {results['metrics']['absolute_accuracy']:6.2f}% "
          f"({results['detailed_stats']['correct_pieces']}/{results['detailed_stats']['total_pieces']} pieces)")
    print(f"Spatial Relationship Accuracy: {results['metrics']['spatial_relationship_accuracy']:6.2f}% "
          f"({results['detailed_stats']['spatial_relationships']['preserved_relationships']}/"
          f"{results['detailed_stats']['spatial_relationships']['total_relationships']} relationships)")
    
    # Print per-relationship-type accuracies
    print("\n" + "-" * 60)
    print("SPATIAL RELATIONSHIPS BY TYPE")
    print("-" * 60)
    for rel_type in ['right', 'down', 'left', 'up']:
        acc = results['detailed_stats']['spatial_relationships']['relationship_accuracies'][rel_type]
        preserved = results['detailed_stats']['spatial_relationships']['relationship_preserved'][rel_type]
        total = results['detailed_stats']['spatial_relationships']['relationship_counts'][rel_type]
        print(f"  {rel_type:>5}: {acc:6.2f}% ({preserved}/{total})")
    print("=" * 60)
    
    # Additional statistics
    per_sample_piece = np.array(results['legacy_format']['per_sample_piece_acc'])
    
    print("\nPer-sample statistics:")
    print(f"  Piece Accuracy:    {per_sample_piece.mean():.4f} ± {per_sample_piece.std():.4f}")
    print(f"  Min/Max Piece Acc: {per_sample_piece.min():.4f} / {per_sample_piece.max():.4f}")
    
    # Load timing info if available
    info_path = args.predictions.replace('.npy', '_info.npz')
    if os.path.exists(info_path):
        info = np.load(info_path)
        print(f"\nTiming information:")
        print(f"  Mean time per puzzle: {info['times'].mean():.2f}s ± {info['times'].std():.2f}s")
        print(f"  Total time: {info['times'].sum():.2f}s")
        if 'scores' in info:
            print(f"  Mean fitness score: {info['scores'].mean():.4f} ± {info['scores'].std():.4f}")
    
    # Save detailed metrics if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving detailed metrics to: {args.output}")
        # Save as JSON for comprehensive results
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save legacy format for backward compatibility
        np.savez(args.output, **results['legacy_format'])


if __name__ == '__main__':
    main()
