"""
Checkpoint management utilities.
"""

import torch
import os
from pathlib import Path
from typing import Dict, Optional


def save_checkpoint(
    state: Dict,
    is_best: bool = False,
    checkpoint_dir: Path = Path('./checkpoints'),
    filename: str = 'checkpoint.pt'
):
    """
    Save model checkpoint.
    
    Args:
        state: Dictionary containing model state, optimizer state, etc.
        is_best: If True, also save as 'best_model.pt'
        checkpoint_dir: Directory to save checkpoint
        filename: Filename for checkpoint
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    print(f"Checkpoint saved: {filepath}")
    
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(state, best_path)
        print(f"Best model saved: {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load model on
        
    Returns:
        checkpoint: Full checkpoint dictionary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Best val accuracy: {checkpoint.get('best_val_accuracy', 'unknown')}")
    
    return checkpoint
