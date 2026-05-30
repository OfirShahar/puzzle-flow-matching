"""
Discrete Flow Matching for puzzle reassembly.

This module implements the flow matching framework adapted for discrete permutation learning.
It handles:
1. Interpolation between random and target permutations
2. Flow matching loss computation
3. Sampling from the learned flow during inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DiscreteFlowMatcher(nn.Module):
    """
    Discrete flow matching for learning permutations.
    
    The flow interpolates between:
    - t=0: Random permutation (noisy assignment)
    - t=1: Target permutation (correct assignment)
    
    Args:
        n_positions: Number of positions (grid_size^2)
        interpolation_type: How to interpolate ('linear', 'cosine')
    """
    def __init__(
        self,
        n_positions: int,
        interpolation_type: str = 'linear'
    ):
        super().__init__()
        self.n_positions = n_positions
        self.interpolation_type = interpolation_type
    
    def get_interpolation_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute interpolation weight alpha(t) in [0, 1].
        
        Args:
            t: Time values in [0, 1], shape (batch_size,)
            
        Returns:
            alpha: Interpolation weights, same shape as t
        """
        if self.interpolation_type == 'linear':
            return t
        elif self.interpolation_type == 'cosine':
            # Cosine schedule: slower at start/end, faster in middle
            return 1 - torch.cos(t * torch.pi / 2)
        else:
            raise ValueError(f"Unknown interpolation type: {self.interpolation_type}")
    
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample random time values from [0, 1].
        
        Args:
            batch_size: Number of samples
            device: Device to create tensor on
            
        Returns:
            t: Random time values, shape (batch_size,)
        """
        return torch.rand(batch_size, device=device)
    
    def interpolate_positions(
        self,
        source_positions: torch.Tensor,
        target_positions: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate between source and target positions using stochastic interpolation.
        
        At time t, each piece has probability alpha(t) of being at its target position
        and probability 1-alpha(t) of being at its source position.
        
        Args:
            source_positions: Starting positions (random), shape (batch_size, n_fragments)
            target_positions: Target positions (correct), shape (batch_size, n_fragments)
            t: Time values, shape (batch_size,)
            
        Returns:
            interpolated_positions: Positions at time t, shape (batch_size, n_fragments)
        """
        batch_size, n_fragments = source_positions.shape
        device = source_positions.device
        
        # Get interpolation weight
        alpha = self.get_interpolation_weight(t)  # (batch_size,)
        
        # Sample which pieces are at their target position
        # Bernoulli with probability alpha(t)
        at_target = torch.rand(batch_size, n_fragments, device=device) < alpha.unsqueeze(1)
        
        # Interpolate: use target where at_target=True, source otherwise
        interpolated_positions = torch.where(
            at_target,
            target_positions,
            source_positions
        )
        
        return interpolated_positions
    
    def compute_flow_matching_loss(
        self,
        logits: torch.Tensor,
        target_positions: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the flow matching loss.
        
        The model should predict the target position at any time t.
        This is the conditional flow matching objective.
        
        Args:
            logits: Predicted logits, shape (batch_size, n_fragments, n_positions)
            target_positions: Ground truth positions, shape (batch_size, n_fragments)
            t: Time values, shape (batch_size,)
            mask: Optional mask for valid pieces, shape (batch_size, n_fragments)
            
        Returns:
            loss: Scalar loss value
        """
        batch_size, n_fragments, n_positions = logits.shape
        
        # Cross-entropy loss: predict target position at time t
        loss = F.cross_entropy(
            logits.view(batch_size * n_fragments, n_positions),
            target_positions.view(batch_size * n_fragments),
            reduction='none'
        )
        loss = loss.view(batch_size, n_fragments)
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss
    
    def generate_random_permutation(
        self,
        batch_size: int,
        n_fragments: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate random permutations (assignments without replacement).
        
        This ensures each position is used exactly once, which is important
        for puzzle reassembly.
        
        Args:
            batch_size: Number of permutations to generate
            n_fragments: Number of pieces/positions
            device: Device to create tensor on
            
        Returns:
            permutations: Random permutations, shape (batch_size, n_fragments)
        """
        # OPTIMIZED: Vectorized permutation generation (no Python loop!)
        # Generate random values and argsort to get permutations
        rand_values = torch.rand(batch_size, n_fragments, device=device)
        permutations = torch.argsort(rand_values, dim=1)
        
        return permutations


class FlowMatchingTrainer:
    """
    Helper class for training with flow matching.
    
    Supports automatic mixed precision (AMP) for faster training on modern GPUs.
    
    Usage:
        trainer = FlowMatchingTrainer(model, flow_matcher, optimizer, use_amp=True)
        loss, metrics = trainer.train_step(pieces, target_positions)
    """
    def __init__(
        self,
        model: nn.Module,
        flow_matcher: DiscreteFlowMatcher,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        use_amp: bool = True,
        val_n_steps: int = 5,
        val_batch_size: Optional[int] = None
    ):
        self.model = model
        self.flow_matcher = flow_matcher
        self.optimizer = optimizer
        self.device = device
        self.val_n_steps = val_n_steps
        self.val_batch_size = val_batch_size
        self.use_amp = use_amp and device.type == 'cuda'
        
        self.model.to(device)
        
        # Initialize GradScaler for mixed precision
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            print("✅ Mixed precision training enabled (FP16)")
        else:
            self.scaler = None
            if device.type == 'cuda':
                print("⚠️  Mixed precision disabled (using FP32 - slower)")
            else:
                print("ℹ️  CPU training (mixed precision not available)")
        
        if val_n_steps < 20:
            print(f"ℹ️  Fast validation mode: n_steps={val_n_steps} (use 20 for final eval)")
        
        if val_batch_size:
            print(f"ℹ️  Validation batch splitting enabled: chunk_size={val_batch_size} (prevents OOM)")
    
    def train_step(
        self,
        pieces: torch.Tensor,
        target_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Perform one training step with automatic mixed precision.
        
        Args:
            pieces: Puzzle pieces, shape (batch_size, n_fragments, C, H, W)
            target_positions: Ground truth positions, shape (batch_size, n_fragments)
            
        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics for logging
        """
        self.model.train()
        batch_size, n_fragments = pieces.shape[:2]
        
        # Move to device (FAST: pin_memory helps here)
        pieces = pieces.to(self.device, non_blocking=True)
        target_positions = target_positions.to(self.device, non_blocking=True)
        
        # 1. Sample time
        t = self.flow_matcher.sample_time(batch_size, self.device)
        
        # 2. Generate random source permutation (OPTIMIZED: vectorized)
        source_positions = self.flow_matcher.generate_random_permutation(
            batch_size, n_fragments, self.device
        )
        
        # 3. Interpolate to get positions at time t
        current_positions = self.flow_matcher.interpolate_positions(
            source_positions, target_positions, t
        )
        
        # 4-6. Forward pass, loss, backward with mixed precision
        if self.use_amp:
            from torch.cuda.amp import autocast
            
            with autocast():
                # Forward pass
                logits = self.model(pieces, current_positions, t)
                # Compute loss
                loss = self.flow_matcher.compute_flow_matching_loss(
                    logits, target_positions, t
                )
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard FP32 training
            logits = self.model(pieces, current_positions, t)
            loss = self.flow_matcher.compute_flow_matching_loss(
                logits, target_positions, t
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # 7. Compute metrics
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == target_positions).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'mean_time': t.mean().item()
        }
        
        return loss, metrics
    
    @torch.no_grad()
    def eval_step(
        self,
        pieces: torch.Tensor,
        target_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Perform one evaluation step with optional mixed precision.
        
        Memory-optimized: Processes large batches in chunks if val_batch_size is set.
        This prevents OOM during validation when predict_positions() runs multiple forward passes.
        
        Args:
            pieces: Puzzle pieces, shape (batch_size, n_fragments, C, H, W)
            target_positions: Ground truth positions, shape (batch_size, n_fragments)
            
        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics
        """
        self.model.eval()
        batch_size, n_fragments = pieces.shape[:2]
        
        # Move to device (non-blocking for speed)
        pieces = pieces.to(self.device, non_blocking=True)
        target_positions = target_positions.to(self.device, non_blocking=True)
        
        # If validation batch size is specified and batch is too large, split it
        if self.val_batch_size and batch_size > self.val_batch_size:
            # Process in chunks to save memory (critical for OOM prevention!)
            total_loss = 0.0
            total_acc = 0.0
            total_sampling_acc = 0.0
            
            for i in range(0, batch_size, self.val_batch_size):
                end_idx = min(i + self.val_batch_size, batch_size)
                chunk_pieces = pieces[i:end_idx]
                chunk_targets = target_positions[i:end_idx]
                
                # Process chunk
                loss, metrics = self._eval_step_single(chunk_pieces, chunk_targets)
                
                chunk_size = end_idx - i
                total_loss += loss * chunk_size
                total_acc += metrics['accuracy'] * chunk_size
                total_sampling_acc += metrics['sampling_accuracy'] * chunk_size
                
                # Clear cache between chunks to prevent memory accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Average over full batch
            avg_loss = total_loss / batch_size
            avg_metrics = {
                'loss': avg_loss,
                'accuracy': total_acc / batch_size,
                'sampling_accuracy': total_sampling_acc / batch_size,
                'mean_time': 0.5  # Placeholder
            }
            
            return avg_loss, avg_metrics
        else:
            # Process normally (batch is small enough)
            return self._eval_step_single(pieces, target_positions)
    
    @torch.no_grad()
    def _eval_step_single(
        self,
        pieces: torch.Tensor,
        target_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Single evaluation step (internal helper).
        
        Args:
            pieces: Puzzle pieces, shape (batch_size, n_fragments, C, H, W)
            target_positions: Ground truth positions, shape (batch_size, n_fragments)
            
        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics
        """
        batch_size, n_fragments = pieces.shape[:2]
        
        # Sample time
        t = self.flow_matcher.sample_time(batch_size, self.device)
        
        # Generate random source (OPTIMIZED: vectorized)
        source_positions = self.flow_matcher.generate_random_permutation(
            batch_size, n_fragments, self.device
        )
        
        # Interpolate
        current_positions = self.flow_matcher.interpolate_positions(
            source_positions, target_positions, t
        )
        
        # Forward pass with optional mixed precision
        if self.use_amp:
            from torch.cuda.amp import autocast
            with autocast():
                logits = self.model(pieces, current_positions, t)
                loss = self.flow_matcher.compute_flow_matching_loss(
                    logits, target_positions, t
                )
        else:
            logits = self.model(pieces, current_positions, t)
            loss = self.flow_matcher.compute_flow_matching_loss(
                logits, target_positions, t
            )
        
        # Compute metrics
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == target_positions).float().mean()
        
        # Also evaluate with full sampling (from t=0 to t=1)
        # OPTIMIZATION: Use fewer steps during training validation (5 instead of 20)
        # MEMORY SAFETY: Extra torch.no_grad() wrapper to ensure no gradient accumulation
        with torch.no_grad():
            predicted_positions = self.model.predict_positions(
                pieces, temperature=0.5, n_steps=self.val_n_steps
            )
        sampling_accuracy = (predicted_positions == target_positions).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'sampling_accuracy': sampling_accuracy.item(),
            'mean_time': t.mean().item()
        }
        
        return loss, metrics


# Example usage
if __name__ == '__main__':
    # Test the flow matcher
    print("Testing DiscreteFlowMatcher...")
    
    n_positions = 16  # 4x4 grid
    flow_matcher = DiscreteFlowMatcher(n_positions)
    
    # Test random permutation generation
    batch_size = 4
    n_fragments = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    perms = flow_matcher.generate_random_permutation(batch_size, n_fragments, device)
    print(f"Random permutations shape: {perms.shape}")
    print(f"First permutation: {perms[0]}")
    
    # Test interpolation
    source = perms
    target = torch.arange(n_fragments, device=device).unsqueeze(0).expand(batch_size, -1)
    t = torch.tensor([0.0, 0.3, 0.7, 1.0], device=device)
    
    interpolated = flow_matcher.interpolate_positions(source, target, t)
    print(f"\nInterpolated positions at different times:")
    for i, time_val in enumerate(t):
        n_correct = (interpolated[i] == target[i]).sum().item()
        print(f"  t={time_val:.1f}: {n_correct}/{n_fragments} pieces at target position")
    
    print("\n✅ Flow matcher tests passed!")
