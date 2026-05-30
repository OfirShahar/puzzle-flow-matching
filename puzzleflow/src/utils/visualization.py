"""
Visualization utilities for puzzle reassembly.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional, Tuple
import os


def visualize_puzzle(
    pieces: torch.Tensor,
    positions: torch.Tensor,
    grid_size: int,
    save_path: Optional[str] = None,
    title: str = "Puzzle"
):
    """
    Visualize assembled puzzle.
    
    Args:
        pieces: (n_fragments, C, H, W) puzzle pieces
        positions: (n_fragments,) position indices or (n_fragments, 2) coordinates
        grid_size: Size of the grid
        save_path: Optional path to save figure
        title: Title for the plot
    """
    n_fragments, C, H, W = pieces.shape
    
    # Convert to numpy and transpose to (H, W, C)
    pieces_np = pieces.cpu().numpy().transpose(0, 2, 3, 1)
    
    # Normalize to [0, 1] if needed
    if pieces_np.max() > 1.0:
        pieces_np = pieces_np / 255.0
    
    # Convert positions to coordinates if needed
    if positions.dim() == 1:
        pos_coords = torch.stack([
            positions // grid_size,
            positions % grid_size
        ], dim=-1).cpu().numpy()
    else:
        pos_coords = positions.cpu().numpy()
    
    # Create canvas
    canvas = np.ones((grid_size * H, grid_size * W, C))
    
    # Place pieces
    for i, (row, col) in enumerate(pos_coords):
        row, col = int(row), int(col)
        if 0 <= row < grid_size and 0 <= col < grid_size:
            canvas[row*H:(row+1)*H, col*W:(col+1)*W] = pieces_np[i]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas)
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    # Draw grid
    for i in range(grid_size + 1):
        ax.axhline(i * H, color='red', linewidth=1, alpha=0.5)
        ax.axvline(i * W, color='red', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def visualize_flow_process(
    model: torch.nn.Module,
    pieces: torch.Tensor,
    target_positions: torch.Tensor,
    grid_size: int,
    n_steps: int = 5,
    save_path: Optional[str] = None
):
    """
    Visualize the flow matching process over time.
    
    Args:
        model: Trained model
        pieces: (1, n_fragments, C, H, W) single puzzle
        target_positions: (1, n_fragments) ground truth positions
        grid_size: Size of the grid
        n_steps: Number of timesteps to visualize
        save_path: Optional path to save figure
    """
    device = pieces.device
    batch_size, n_fragments = 1, pieces.shape[1]
    
    # Generate timesteps
    timesteps = torch.linspace(0, 1, n_steps, device=device)
    
    # Start from random positions
    current_positions = torch.randperm(n_fragments, device=device).unsqueeze(0)
    
    # Create subplot
    fig, axes = plt.subplots(1, n_steps, figsize=(4*n_steps, 4))
    
    for idx, t in enumerate(timesteps):
        # Get predictions at time t
        t_batch = torch.ones(batch_size, device=device) * t
        logits = model(pieces, current_positions, t_batch)
        
        # Update positions
        probs = torch.softmax(logits, dim=-1)
        current_positions = torch.multinomial(probs.squeeze(0), 1).squeeze(-1).unsqueeze(0)
        
        # Visualize
        pieces_np = pieces[0].cpu().numpy().transpose(0, 2, 3, 1)
        if pieces_np.max() > 1.0:
            pieces_np = pieces_np / 255.0
        
        pos_coords = torch.stack([
            current_positions[0] // grid_size,
            current_positions[0] % grid_size
        ], dim=-1).cpu().numpy()
        
        H, W = pieces_np.shape[1:3]
        C = pieces_np.shape[-1]
        canvas = np.ones((grid_size * H, grid_size * W, C))
        
        for i, (row, col) in enumerate(pos_coords):
            row, col = int(row), int(col)
            if 0 <= row < grid_size and 0 <= col < grid_size:
                canvas[row*H:(row+1)*H, col*W:(col+1)*W] = pieces_np[i]
        
        axes[idx].imshow(canvas)
        axes[idx].set_title(f't={t:.2f}', fontsize=12)
        axes[idx].axis('off')
        
        # Draw grid
        for i in range(grid_size + 1):
            axes[idx].axhline(i * H, color='red', linewidth=0.5, alpha=0.3)
            axes[idx].axvline(i * W, color='red', linewidth=0.5, alpha=0.3)
    
    plt.suptitle('Flow Matching Process', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved flow visualization to: {save_path}")
    
    plt.show()


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    save_path: Optional[str] = None
):
    """
    Plot training curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train', linewidth=2)
    ax1.plot(val_losses, label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train', linewidth=2)
    ax2.plot(val_accs, label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to: {save_path}")
    
    plt.show()
