"""
Prediction script for ViT-based Puzzle Flow Matching Model.

This script runs predictions on a dataset and saves the results in a standard format
for evaluation with custom metrics.
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    PuzzleFlowViT, 
    PuzzleFlowViT_V2,
    PuzzleFlowViT_DirectPrediction,
    PuzzleFlowViT_CustomSchedule,
    PuzzleFlowViT_VariableLayers
)
from src.utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for ViT-based puzzle model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to predict on')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path for predictions (will save as .npz)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for prediction (default: 4, use 1-2 for large models)')
    parser.add_argument('--n-steps', type=int, default=20,
                        help='Number of inference steps for flow matching (higher=better but slower)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--use-v2', action='store_true',
                        help='Use ViT V2 architecture (with cross-piece attention)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Try to get config from checkpoint, or load from YAML file
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("Config loaded from checkpoint")
    else:
        # Look for config.yaml in the same directory as the checkpoint
        checkpoint_path = Path(args.checkpoint)
        config_path = checkpoint_path.parent / 'config.yaml'
        
        if config_path.exists():
            print(f"Config not in checkpoint, loading from: {config_path}")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(
                f"Config not found in checkpoint and config.yaml not found at: {config_path}\n"
                f"Please ensure config.yaml exists alongside the checkpoint file."
            )
    
    # Load dataset first to detect channels
    from src.data import PuzzleDataset
    from torch.utils.data import DataLoader
    
    print(f"\nLoading {args.split} dataset from: {args.data_root}")
    
    dataset = PuzzleDataset(
        data_dir=f"{args.data_root}/{args.split}",
        use_coordinates=False,
        normalize=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Detect number of channels and grid size from dataset
    sample_pieces, _ = next(iter(dataloader))
    n_channels = sample_pieces.shape[2]  # (B, N, C, H, W)
    n_fragments = sample_pieces.shape[1]  # Number of pieces
    actual_grid_size = int(np.sqrt(n_fragments))
    
    print(f"Detected {n_channels} input channels")
    print(f"Detected {n_fragments} pieces ({actual_grid_size}×{actual_grid_size} grid)")
    
    # Validate grid size
    if actual_grid_size * actual_grid_size != n_fragments:
        raise ValueError(f"Number of pieces ({n_fragments}) is not a perfect square!")
    
    # Check if we should use V2 architecture
    use_cross_piece = args.use_v2 or config.get('use_cross_piece_attention', False)
    
    # Determine model type from config
    model_type = config.get('model_type', 'standard')
    ablation_type = config.get('ablation_type', None)
    
    # Create model with detected grid size
    print("Creating ViT-based model...")
    
    # Check for ablation models
    if model_type == 'direct_prediction' or ablation_type == 'direct_prediction':
        print("Using Direct Prediction model (no flow matching)")
        model = PuzzleFlowViT_DirectPrediction(
            piece_size=tuple(config['piece_size']),
            grid_size=actual_grid_size,
            in_channels=n_channels,
            vit_model_name=config['vit_model_name'],
            freeze_vit=config.get('freeze_vit', False),
            freeze_layers=config.get('freeze_layers', 0),
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout']
        )
        is_direct_prediction = True
    elif model_type == 'custom_schedule' or ablation_type in ['schedule_linear', 'schedule_cos', 'schedule_sqrt', 'schedule_cosine']:
        print(f"Using Custom Schedule model: {ablation_type}")
        # Map config interpolation_type to schedule_type parameter
        interpolation_type = config.get('interpolation_type', 'linear')
        # Map cosine to schedule_type expected values
        if interpolation_type == 'cosine':
            schedule_type = 'cosine'
        elif interpolation_type == 'sqrt':
            schedule_type = 'quadratic'  # sqrt uses quadratic schedule
        else:
            schedule_type = interpolation_type
            
        model = PuzzleFlowViT_CustomSchedule(
            piece_size=tuple(config['piece_size']),
            grid_size=actual_grid_size,
            in_channels=n_channels,
            vit_model_name=config['vit_model_name'],
            freeze_vit=config.get('freeze_vit', False),
            freeze_layers=config.get('freeze_layers', 0),
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            schedule_type=schedule_type
        )
        is_direct_prediction = False
    elif model_type == 'variable_layers' or ablation_type in ['layers_0', 'layers_8']:
        print(f"Using Variable Layers model: {config['n_layers']} layers")
        model = PuzzleFlowViT_VariableLayers(
            piece_size=tuple(config['piece_size']),
            grid_size=actual_grid_size,
            in_channels=n_channels,
            vit_model_name=config['vit_model_name'],
            freeze_vit=config.get('freeze_vit', False),
            freeze_layers=config.get('freeze_layers', 0),
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout']
        )
        is_direct_prediction = False
    elif use_cross_piece:
        print("Using ViT V2 architecture (with cross-piece attention)")
        model = PuzzleFlowViT_V2(
            piece_size=tuple(config['piece_size']),
            grid_size=actual_grid_size,
            in_channels=n_channels,
            vit_model_name=config['vit_model_name'],
            freeze_vit=config.get('freeze_vit', False),
            freeze_layers=config.get('freeze_layers', 0),
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            use_cross_piece_attention=True
        )
        is_direct_prediction = False
    else:
        print("Using ViT V1 architecture (independent piece processing)")
        model = PuzzleFlowViT(
            piece_size=tuple(config['piece_size']),
            grid_size=actual_grid_size,
            in_channels=n_channels,
            vit_model_name=config['vit_model_name'],
            freeze_vit=config.get('freeze_vit', False),
            freeze_layers=config.get('freeze_layers', 0),
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout']
        )
        is_direct_prediction = False
    
    if is_direct_prediction:
        print("⚠️  Direct prediction model detected - will use n_steps=1 for inference")
    
    # Load model weights - use strict=False for direct prediction to ignore time_mlp layers
    if is_direct_prediction:
        # Filter out time_mlp keys that don't exist in DirectPrediction model
        state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if not k.startswith('time_mlp')}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"⚠️  Filtered out unexpected keys: {unexpected}")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded (epoch {checkpoint['epoch']}, best val acc: {checkpoint['best_val_accuracy']:.3f})")
    
    # Override n_steps for direct prediction models
    if is_direct_prediction:
        actual_n_steps = 1
        print("⚠️  Overriding n_steps to 1 for direct prediction model")
    else:
        actual_n_steps = args.n_steps
    
    # Run predictions
    print(f"\n{'='*60}")
    print(f"Running predictions on {len(dataset)} samples...")
    print(f"Using {actual_n_steps} inference steps")
    print(f"{'='*60}\n")
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for pieces, target_positions in tqdm(dataloader, desc='Predicting'):
            pieces = pieces.to(device)
            target_positions = target_positions.to(device)
            
            # Convert coordinates to indices if needed
            if target_positions.dim() == 3:
                target_positions = target_positions[:, :, 0] * actual_grid_size + target_positions[:, :, 1]
            
            # Predict positions (different signature for direct prediction models)
            if is_direct_prediction:
                predicted_positions = model.predict_positions(pieces, temperature=0.5)
            else:
                predicted_positions = model.predict_positions(pieces, temperature=0.5, n_steps=actual_n_steps)
            
            # Store predictions and targets
            all_predictions.append(predicted_positions.cpu().numpy())
            all_targets.append(target_positions.cpu().numpy())
            
            # Clear cache to prevent memory fragmentation
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)  # (N, num_pieces)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, num_pieces)
    
    print(f"\n{'='*60}")
    print(f"Predictions shape: {all_predictions.shape}")
    print(f"Targets shape: {all_targets.shape}")
    
    # Quick accuracy check
    position_accuracy = (all_predictions == all_targets).mean()
    exact_match_accuracy = (all_predictions == all_targets).all(axis=1).mean()
    
    print(f"\nQuick metrics:")
    print(f"  Position accuracy: {position_accuracy:.4f} ({position_accuracy*100:.2f}%)")
    print(f"  Exact match accuracy: {exact_match_accuracy:.4f} ({exact_match_accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as compressed numpy file
    np.savez_compressed(
        output_path,
        predictions=all_predictions,
        targets=all_targets,
        grid_size=actual_grid_size,
        dataset=args.data_root,
        split=args.split,
        checkpoint=args.checkpoint,
        n_steps=args.n_steps,
        model_type='puzzleflow_vit_v2' if use_cross_piece else 'puzzleflow_vit_v1'
    )
    
    print(f"\n✅ Predictions saved to: {output_path}")
    print(f"   Format: .npz file with keys: predictions, targets, grid_size, metadata")
    print(f"   Load with: data = np.load('{output_path}')")
    print(f"             predictions = data['predictions']")
    print(f"             targets = data['targets']")
    
    # Also save a JSON metadata file for easy inspection
    metadata = {
        'dataset': args.data_root,
        'split': args.split,
        'checkpoint': args.checkpoint,
        'n_samples': int(len(all_predictions)),
        'n_pieces': int(all_predictions.shape[1]),
        'grid_size': int(actual_grid_size),
        'n_steps': args.n_steps,
        'model_type': 'puzzleflow_vit_v2' if use_cross_piece else 'puzzleflow_vit_v1',
        'quick_metrics': {
            'position_accuracy': float(position_accuracy),
            'exact_match_accuracy': float(exact_match_accuracy),
        }
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to: {metadata_path}")
    print()


if __name__ == '__main__':
    main()
