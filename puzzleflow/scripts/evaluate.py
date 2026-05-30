"""
Evaluation script for ViT-based Puzzle Flow Matching Model.
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import PuzzleFlowViT
from src.training.metrics import compute_puzzle_metrics
from src.utils.checkpoint import load_checkpoint


def evaluate_model(model, dataloader, device, grid_size, n_steps=20, save_predictions=True):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to use
        grid_size: Grid size
        n_steps: Number of inference steps for flow matching
        save_predictions: Whether to save predictions for later analysis
        
    Returns:
        metrics: Dictionary of evaluation metrics
        predictions_data: Dictionary containing predictions and targets (if save_predictions=True)
    """
    model.eval()
    
    all_position_accs = []
    all_exact_match_accs = []
    all_neighbor_accs = []
    
    # For saving predictions
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for pieces, target_positions in tqdm(dataloader, desc='Evaluating'):
            pieces = pieces.to(device)
            target_positions = target_positions.to(device)
            
            # Convert coordinates to indices if needed
            if target_positions.dim() == 3:
                target_positions = target_positions[:, :, 0] * grid_size + target_positions[:, :, 1]
            
            # Predict positions
            predicted_positions = model.predict_positions(pieces, temperature=0.5, n_steps=n_steps)
            
            # Compute metrics
            metrics = compute_puzzle_metrics(predicted_positions, target_positions, grid_size)
            
            all_position_accs.append(metrics['position_accuracy'])
            all_exact_match_accs.append(metrics['exact_match_accuracy'])
            all_neighbor_accs.append(metrics['neighbor_accuracy'])
            
            # Save predictions if requested
            if save_predictions:
                all_predictions.append(predicted_positions.cpu().numpy())
                all_targets.append(target_positions.cpu().numpy())
    
    # Average metrics
    avg_metrics = {
        'position_accuracy': sum(all_position_accs) / len(all_position_accs),
        'exact_match_accuracy': sum(all_exact_match_accs) / len(all_exact_match_accs),
        'neighbor_accuracy': sum(all_neighbor_accs) / len(all_neighbor_accs),
        'num_samples': len(dataloader.dataset)
    }
    
    # Prepare predictions data
    predictions_data = None
    if save_predictions:
        predictions_data = {
            'predictions': np.concatenate(all_predictions, axis=0),  # (N, num_pieces)
            'targets': np.concatenate(all_targets, axis=0)  # (N, num_pieces)
        }
    
    return avg_metrics, predictions_data


def main():
    parser = argparse.ArgumentParser(description='Evaluate ViT-based Puzzle Flow Matching Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--n-steps', type=int, default=20,
                        help='Number of inference steps for flow matching (higher=better but slower)')
    parser.add_argument('--output', type=str, default='vit_evaluation_results.json',
                        help='Path to save evaluation results')
    parser.add_argument('--save-predictions', action='store_true', default=True,
                        help='Save predictions for later analysis (default: True)')
    parser.add_argument('--no-save-predictions', action='store_false', dest='save_predictions',
                        help='Do not save predictions')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
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
    config = checkpoint['config']
    
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
        num_workers=0,
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
    
    # Create model with detected grid size
    print("Creating ViT-based model...")
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Model loaded (epoch {checkpoint['epoch']}, best val acc: {checkpoint['best_val_accuracy']:.3f})")
    
    # Evaluate
    print("\n" + "="*60)
    print("Starting evaluation...")
    print(f"Using {args.n_steps} inference steps")
    print("="*60 + "\n")
    
    metrics, predictions_data = evaluate_model(
        model, dataloader, device, actual_grid_size, 
        n_steps=args.n_steps,
        save_predictions=args.save_predictions
    )
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\n📊 Piece-Level Metrics:")
    print(f"  Position Accuracy:     {metrics['position_accuracy']:.4f} ({metrics['position_accuracy']*100:.2f}%)")
    print(f"    → Average pieces correct per puzzle: {metrics['position_accuracy'] * actual_grid_size**2:.2f}/{actual_grid_size**2}")
    print(f"\n🎯 Puzzle-Level Metrics:")
    print(f"  Exact Match Accuracy:  {metrics['exact_match_accuracy']:.4f} ({metrics['exact_match_accuracy']*100:.2f}%)")
    print(f"    → Percentage of fully solved puzzles")
    print(f"\n🔗 Neighbor Metrics:")
    print(f"  Neighbor Accuracy:     {metrics['neighbor_accuracy']:.4f} ({metrics['neighbor_accuracy']*100:.2f}%)")
    print(f"    → Percentage of correct neighbor relationships")
    print(f"\n📈 Dataset Statistics:")
    print(f"  Number of Samples:     {metrics['num_samples']}")
    print(f"  Grid Size:             {actual_grid_size}×{actual_grid_size}")
    print(f"  Inference Steps:       {args.n_steps}")
    print("="*70)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'n_steps': args.n_steps,
        'metrics': metrics,
        'config': config
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Save predictions if requested
    if args.save_predictions and predictions_data is not None:
        predictions_path = output_path.with_name(output_path.stem + '_predictions.npz')
        np.savez_compressed(
            predictions_path,
            predictions=predictions_data['predictions'],
            targets=predictions_data['targets'],
            grid_size=actual_grid_size
        )
        print(f"✅ Predictions saved to: {predictions_path}")
        print(f"   Shape: {predictions_data['predictions'].shape}")
        print(f"   Use: np.load('{predictions_path}') to load predictions")


if __name__ == '__main__':
    main()
