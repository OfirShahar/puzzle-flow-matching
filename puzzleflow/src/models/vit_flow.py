"""
Vision Transformer (ViT) based model for puzzle reassembly using Flow Matching.

This model uses a pretrained ViT encoder from HuggingFace to extract visual features,
then applies flow matching to learn the permutation of puzzle pieces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import math


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for continuous values."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, ...) tensor of positions
            
        Returns:
            embeddings: (..., dim) sinusoidal embeddings
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class PuzzleFlowViT(nn.Module):
    """
    ViT-based Puzzle Flow Matching Model.
    
    Uses a pretrained Vision Transformer to extract features from puzzle pieces,
    then applies flow matching to predict piece positions.
    """
    
    def __init__(
        self,
        piece_size: tuple = (96, 96),
        grid_size: int = 3,
        in_channels: int = 3,
        vit_model_name: str = 'google/vit-base-patch16-224',
        freeze_vit: bool = False,
        freeze_layers: int = 0,
        d_model: int = 768,  # ViT-base output dim
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Args:
            piece_size: (H, W) size of each puzzle piece
            grid_size: Size of puzzle grid (e.g., 3 for 3×3)
            in_channels: Number of input channels (3 for RGB, 4 for RGBA)
            vit_model_name: HuggingFace ViT model name
            freeze_vit: If True, freeze entire ViT backbone
            freeze_layers: Number of ViT layers to freeze (0 = none, 12 = all for base)
            d_model: Dimension of transformer
            n_heads: Number of attention heads
            n_layers: Number of additional transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.piece_size = piece_size
        self.grid_size = grid_size
        self.n_positions = grid_size * grid_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Load pretrained ViT
        print(f"Loading pretrained ViT: {vit_model_name}")
        self.vit = ViTModel.from_pretrained(vit_model_name)
        
        # Enable gradient checkpointing to save memory (critical for large batches!)
        if hasattr(self.vit, 'gradient_checkpointing_enable'):
            self.vit.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled (saves ~30% memory)")
        
        # Get ViT output dimension
        vit_dim = self.vit.config.hidden_size
        
        # Handle different input channels
        if in_channels != 3:
            print(f"Adapting ViT for {in_channels} channels...")
            # Add a projection layer to convert to 3 channels
            self.channel_adapter = nn.Conv2d(in_channels, 3, 1)
        else:
            self.channel_adapter = None
        
        # Resize images to ViT input size if needed
        self.vit_input_size = self.vit.config.image_size  # Usually 224
        
        # Freeze ViT layers if requested
        if freeze_vit:
            print("Freezing entire ViT backbone")
            for param in self.vit.parameters():
                param.requires_grad = False
        elif freeze_layers > 0:
            print(f"Freezing first {freeze_layers} ViT layers")
            # Freeze embeddings
            for param in self.vit.embeddings.parameters():
                param.requires_grad = False
            # Freeze specified number of encoder layers
            for i in range(min(freeze_layers, len(self.vit.encoder.layer))):
                for param in self.vit.encoder.layer[i].parameters():
                    param.requires_grad = False
        
        # Project ViT features to d_model if different
        if vit_dim != d_model:
            self.vit_proj = nn.Linear(vit_dim, d_model)
        else:
            self.vit_proj = nn.Identity()
        
        # Position embeddings (for current piece positions in flow)
        self.position_embedding = nn.Embedding(self.n_positions, d_model)
        
        # Time embedding (for flow matching time t)
        self.time_embed_dim = d_model // 4
        self.time_embedding = SinusoidalPositionEmbedding(self.time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Additional transformer layers for puzzle-specific reasoning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head to predict position logits
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, self.n_positions)
        )
    
    def encode_pieces(self, pieces: torch.Tensor) -> torch.Tensor:
        """
        Encode puzzle pieces using pretrained ViT.
        
        Args:
            pieces: (batch_size, n_pieces, C, H, W)
            
        Returns:
            features: (batch_size, n_pieces, d_model)
        """
        B, N, C, H, W = pieces.shape
        
        # Flatten batch and pieces dimensions
        pieces_flat = pieces.view(B * N, C, H, W)
        
        # Adapt channels if needed
        if self.channel_adapter is not None:
            pieces_flat = self.channel_adapter(pieces_flat)
        
        # Resize to ViT input size if needed
        if (H, W) != (self.vit_input_size, self.vit_input_size):
            pieces_flat = F.interpolate(
                pieces_flat,
                size=(self.vit_input_size, self.vit_input_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Extract features using ViT (use [CLS] token)
        with torch.set_grad_enabled(not self.vit.training or any(p.requires_grad for p in self.vit.parameters())):
            outputs = self.vit(pieces_flat)
            # Use [CLS] token representation
            features = outputs.last_hidden_state[:, 0]  # (B*N, vit_dim)
        
        # Project to d_model
        features = self.vit_proj(features)  # (B*N, d_model)
        
        # Reshape back
        features = features.view(B, N, self.d_model)
        
        return features
    
    def forward(
        self,
        pieces: torch.Tensor,
        positions: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for flow matching training.
        
        Args:
            pieces: (batch_size, n_pieces, C, H, W) - puzzle pieces
            positions: (batch_size, n_pieces) - current position indices at time t
            t: (batch_size,) - flow time in [0, 1]
            
        Returns:
            logits: (batch_size, n_pieces, n_positions) - predicted position logits
        """
        batch_size, n_pieces = pieces.shape[:2]
        
        # Encode pieces using ViT
        piece_features = self.encode_pieces(pieces)  # (B, N, d_model)
        
        # Add position embeddings
        pos_embed = self.position_embedding(positions)  # (B, N, d_model)
        features = piece_features + pos_embed
        
        # Add time embedding (broadcast to all pieces)
        time_embed = self.time_embedding(t)  # (B, time_embed_dim)
        time_embed = self.time_mlp(time_embed)  # (B, d_model)
        features = features + time_embed.unsqueeze(1)  # (B, N, d_model)
        
        # Apply transformer for piece interactions (skip if n_layers=0)
        if self.n_layers > 0:
            features = self.transformer(features)  # (B, N, d_model)
        
        # Predict position logits
        logits = self.output_head(features)  # (B, N, n_positions)
        
        return logits
    
    def _vectorized_greedy_assignment(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Vectorized greedy assignment ensuring valid permutations.
        
        Uses iterative maximum matching in parallel across the batch.
        NO Python loops over batch dimension, NO .item() calls.
        ALL operations stay on GPU for maximum performance.
        
        Args:
            logits: (batch_size, n_pieces, n_positions) score matrix
            
        Returns:
            assignments: (batch_size, n_pieces) permutation indices
        """
        batch_size, n_pieces, n_positions = logits.shape
        device = logits.device
        
        # Initialize assignment tensor
        assignments = torch.zeros(batch_size, n_pieces, dtype=torch.long, device=device)
        
        # Track which positions are used (batch_size, n_positions)
        used_mask = torch.zeros(batch_size, n_positions, dtype=torch.bool, device=device)
        
        # Greedy assignment: assign pieces one by one (vectorized across batch)
        for piece_idx in range(n_pieces):
            # Get scores for current piece across all batches
            piece_scores = logits[:, piece_idx, :]  # (batch_size, n_positions)
            
            # Mask out already used positions
            piece_scores = piece_scores.masked_fill(used_mask, float('-inf'))
            
            # Find best position for each sample in batch (vectorized!)
            best_positions = torch.argmax(piece_scores, dim=1)  # (batch_size,)
            
            # Assign
            assignments[:, piece_idx] = best_positions
            
            # Update used_mask: set used_mask[b, best_positions[b]] = True
            batch_indices = torch.arange(batch_size, device=device)
            used_mask[batch_indices, best_positions] = True
        
        return assignments
    
    def predict_positions(
        self,
        pieces: torch.Tensor,
        temperature: float = 1.0,
        n_steps: int = 10
    ) -> torch.Tensor:
        """
        Predict final positions at t=1 (inference mode).
        
        OPTIMIZED: Uses vectorized operations, no Python loops over batch dimension.
        
        Args:
            pieces: (batch_size, n_pieces, C, H, W)
            temperature: Temperature for sampling (lower = more confident)
            n_steps: Number of flow steps
            
        Returns:
            positions: (batch_size, n_pieces) - predicted position indices
        """
        batch_size, n_pieces = pieces.shape[:2]
        device = pieces.device
        
        # OPTIMIZED: Vectorized random permutation generation (no list comprehension)
        positions = torch.argsort(
            torch.rand(batch_size, n_pieces, device=device), dim=1
        )
        
        # Evolve through flow with multiple steps
        for step in range(n_steps):
            # OPTIMIZED: Batch time tensor (not scalar)
            t = torch.full((batch_size,), (step + 1) / n_steps, device=device)
            
            logits = self.forward(pieces, positions, t)  # (B, N, n_positions)
            logits = logits / temperature
            
            # OPTIMIZED: Vectorized greedy assignment (no Python loops, no .item() calls)
            positions = self._vectorized_greedy_assignment(logits)
        
        return positions


def create_vit_model(
    piece_size: tuple = (96, 96),
    grid_size: int = 3,
    in_channels: int = 3,
    vit_model_name: str = 'google/vit-base-patch16-224',
    freeze_vit: bool = False,
    **kwargs
) -> PuzzleFlowViT:
    """
    Factory function to create a PuzzleFlowViT model.
    
    Args:
        piece_size: Size of puzzle pieces
        grid_size: Grid dimension
        in_channels: Number of input channels
        vit_model_name: HuggingFace ViT model to use
        freeze_vit: Whether to freeze ViT backbone
        **kwargs: Additional arguments for PuzzleFlowViT
        
    Returns:
        model: Initialized PuzzleFlowViT model
    """
    return PuzzleFlowViT(
        piece_size=piece_size,
        grid_size=grid_size,
        in_channels=in_channels,
        vit_model_name=vit_model_name,
        freeze_vit=freeze_vit,
        **kwargs
    )
