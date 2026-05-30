"""
Loss functions for puzzle flow matching.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def flow_matching_loss(
    logits: torch.Tensor,
    target_positions: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute cross-entropy loss for position prediction.
    
    Args:
        logits: (batch_size, n_fragments, n_positions)
        target_positions: (batch_size, n_fragments)
        mask: Optional (batch_size, n_fragments) binary mask
        
    Returns:
        loss: Scalar loss value
    """
    batch_size, n_fragments, n_positions = logits.shape
    
    loss = F.cross_entropy(
        logits.view(batch_size * n_fragments, n_positions),
        target_positions.view(batch_size * n_fragments),
        reduction='none'
    )
    loss = loss.view(batch_size, n_fragments)
    
    if mask is not None:
        loss = loss * mask
        loss = loss.sum() / mask.sum()
    else:
        loss = loss.mean()
    
    return loss


def accuracy_metric(
    logits: torch.Tensor,
    target_positions: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute accuracy metric.
    
    Args:
        logits: (batch_size, n_fragments, n_positions)
        target_positions: (batch_size, n_fragments)
        mask: Optional (batch_size, n_fragments) binary mask
        
    Returns:
        accuracy: Scalar accuracy value
    """
    predictions = logits.argmax(dim=-1)
    correct = (predictions == target_positions).float()
    
    if mask is not None:
        correct = correct * mask
        accuracy = correct.sum() / mask.sum()
    else:
        accuracy = correct.mean()
    
    return accuracy
