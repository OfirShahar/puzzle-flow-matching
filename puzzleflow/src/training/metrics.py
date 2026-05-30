"""
Evaluation metrics for puzzle reassembly.
"""

import torch
import numpy as np
from typing import Dict, Tuple


def position_accuracy(
    predicted_positions: torch.Tensor,
    target_positions: torch.Tensor
) -> float:
    """
    Compute position accuracy.
    
    Args:
        predicted_positions: (batch_size, n_fragments)
        target_positions: (batch_size, n_fragments)
        
    Returns:
        accuracy: Fraction of correctly placed pieces
    """
    correct = (predicted_positions == target_positions).float()
    return correct.mean().item()


def neighbor_accuracy(
    predicted_positions: torch.Tensor,
    target_positions: torch.Tensor,
    grid_size: int
) -> float:
    """
    Compute neighbor accuracy - how many pieces have correct neighbors.
    
    Args:
        predicted_positions: (batch_size, n_fragments)
        target_positions: (batch_size, n_fragments)
        grid_size: Size of the grid
        
    Returns:
        accuracy: Fraction of correct neighbor relationships
    """
    batch_size, n_fragments = predicted_positions.shape
    
    # Convert indices to coordinates
    pred_coords = torch.stack([
        predicted_positions // grid_size,
        predicted_positions % grid_size
    ], dim=-1)  # (B, N, 2)
    
    target_coords = torch.stack([
        target_positions // grid_size,
        target_positions % grid_size
    ], dim=-1)  # (B, N, 2)
    
    total_correct = 0
    total_neighbors = 0
    
    for b in range(batch_size):
        for i in range(n_fragments):
            for j in range(n_fragments):
                if i == j:
                    continue
                
                # Check if they are neighbors in target
                target_dist = torch.abs(target_coords[b, i] - target_coords[b, j]).sum()
                if target_dist == 1:  # They are neighbors
                    # Check if they are also neighbors in prediction
                    pred_dist = torch.abs(pred_coords[b, i] - pred_coords[b, j]).sum()
                    if pred_dist == 1:
                        total_correct += 1
                    total_neighbors += 1
    
    return total_correct / max(total_neighbors, 1)


def exact_match_accuracy(
    predicted_positions: torch.Tensor,
    target_positions: torch.Tensor
) -> float:
    """
    Compute exact match accuracy (puzzle-level accuracy).
    A puzzle is correct only if ALL pieces are in the correct positions.
    
    Args:
        predicted_positions: (batch_size, n_fragments)
        target_positions: (batch_size, n_fragments)
        
    Returns:
        accuracy: Fraction of puzzles that are completely solved
    """
    # Check if all pieces are correct for each puzzle
    all_correct = (predicted_positions == target_positions).all(dim=1).float()
    return all_correct.mean().item()


def compute_puzzle_metrics(
    predicted_positions: torch.Tensor,
    target_positions: torch.Tensor,
    grid_size: int
) -> Dict[str, float]:
    """
    Compute comprehensive puzzle metrics.
    
    Args:
        predicted_positions: (batch_size, n_fragments)
        target_positions: (batch_size, n_fragments)
        grid_size: Size of the grid
        
    Returns:
        metrics: Dictionary of metrics including:
            - position_accuracy: Piece-level accuracy (% of pieces in correct positions)
            - exact_match_accuracy: Puzzle-level accuracy (% of fully solved puzzles)
            - neighbor_accuracy: % of correct neighbor relationships
    """
    metrics = {
        'position_accuracy': position_accuracy(predicted_positions, target_positions),
        'exact_match_accuracy': exact_match_accuracy(predicted_positions, target_positions),
        'neighbor_accuracy': neighbor_accuracy(predicted_positions, target_positions, grid_size)
    }
    
    return metrics


def permutation_distance(perm1: torch.Tensor, perm2: torch.Tensor) -> float:
    """
    Compute distance between two permutations (number of disagreements).
    
    Args:
        perm1: (batch_size, n_fragments)
        perm2: (batch_size, n_fragments)
        
    Returns:
        distance: Average number of disagreements
    """
    disagreements = (perm1 != perm2).float().sum(dim=1)
    return disagreements.mean().item()
