"""
Enhanced Training script for ViT-based Puzzle Flow Matching Model.

This script adds:
- Early stopping with configurable patience
- Improved checkpoint resumption
- Support for running multiple trials
- Better logging and experiment tracking
"""

import sys
import argparse
import os
from pathlib import Path
import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import PuzzleFlowViT, PuzzleFlowViT_V2
from src.models.flow_matcher import DiscreteFlowMatcher, FlowMatchingTrainer
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.config import load_config, save_config
from src.utils.visualization import visualize_puzzle

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")


class EarlyStopping:
    """Early stopping handler to stop training when validation metric stops improving."""
    
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics to maximize (accuracy), 'min' for metrics to minimize (loss)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


def main():
    parser = argparse.ArgumentParser(description='Train ViT-based Puzzle Flow Matching Model')
    parser.add_argument('--config', type=str, default='configs/vit_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--freeze', action='store_true',
                        help='Freeze ViT backbone (feature extraction only)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of data loading workers')
    parser.add_argument('--val-batch-size', type=int, default=None,
                        help='Validation batch size')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Training batch size (overrides config)')
    parser.add_argument('--val-n-steps', type=int, default=None,
                        help='Number of validation flow steps')
    parser.add_argument('--use-amp', action='store_true',
                        help='Use automatic mixed precision (FP16)')
    parser.add_argument('--compile-model', action='store_true',
                        help='Compile model with torch.compile')
    parser.add_argument('--test-validation', action='store_true',
                        help='Run a quick validation test before training')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Checkpoint directory (overrides config)')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Early stopping patience (default: 10 epochs)')
    parser.add_argument('--run-id', type=str, default='run1',
                        help='Run identifier for multiple trials (e.g., run1, run2, run3)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.freeze:
        config['freeze_vit'] = True
        print("⚠️  Overriding config: freeze_vit = True (from --freeze flag)")
    
    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
        print(f"⚠️  Overriding config: num_workers = {args.num_workers}")
    
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
        print(f"⚠️  Overriding config: batch_size = {args.batch_size}")
    
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
        print(f"⚠️  Overriding config: num_epochs = {args.epochs}")
    
    # Update checkpoint directory to include run_id
    if args.checkpoint_dir is not None:
        base_checkpoint_dir = args.checkpoint_dir
    else:
        base_checkpoint_dir = config['checkpoint_dir']
    
    config['checkpoint_dir'] = str(Path(base_checkpoint_dir) / args.run_id)
    print(f"⚠️  Checkpoint directory: {config['checkpoint_dir']}")
    
    if args.use_amp or config.get('use_amp', False):
        config['use_amp'] = True
        print(f"⚠️  Using mixed precision (AMP) for training")
    
    if args.val_n_steps is not None:
        config['val_n_steps'] = args.val_n_steps
        print(f"⚠️  Overriding config: val_n_steps = {args.val_n_steps}")
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    from src.data import create_dataloaders
    
    print(f"\nLoading dataset from: {args.data_root}")
    
    # Efficiency settings for DataLoader
    num_workers = config['num_workers']
    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None
    
    print(f"DataLoader settings:")
    print(f"  num_workers: {num_workers}")
    print(f"  persistent_workers: {persistent_workers}")
    print(f"  prefetch_factor: {prefetch_factor}")
    print(f"  pin_memory: {config.get('pin_memory', True)}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=config['batch_size'],
        use_coordinates=False,
        num_workers=num_workers,
        normalize=True,
        pin_memory=config.get('pin_memory', True)
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Detect number of channels and grid size from dataset
    sample_pieces, _ = next(iter(train_loader))
    n_channels = sample_pieces.shape[2]
    n_fragments = sample_pieces.shape[1]
    actual_grid_size = int(np.sqrt(n_fragments))
    
    print(f"Detected {n_channels} input channels")
    print(f"Detected {n_fragments} pieces ({actual_grid_size}×{actual_grid_size} grid)")
    
    if actual_grid_size * actual_grid_size != n_fragments:
        raise ValueError(f"Number of pieces ({n_fragments}) is not a perfect square!")
    
    config['grid_size'] = actual_grid_size
    
    # Initialize W&B if API key is provided
    use_wandb = False
    wandb_api_key = config.get('wandb_api_key')
    
    if wandb_api_key is None:
        wandb_api_key = os.environ.get('WANDB_API_KEY')
    
    if wandb_api_key and WANDB_AVAILABLE:
        use_wandb = True
        os.environ['WANDB_API_KEY'] = wandb_api_key
        
        # Add run_id to wandb run name
        run_name = config.get('run_name')
        if run_name:
            run_name = f"{run_name}_{args.run_id}"
        else:
            dataset_name = Path(args.data_root).name
            run_name = f"vit_{dataset_name}_{args.run_id}"
        
        wandb.init(
            project=config.get('project_name', 'puzzle-flow-matching-vit'),
            name=run_name,
            config=config,
            tags=['vit', f'grid_{actual_grid_size}x{actual_grid_size}', 
                  f'channels_{n_channels}', 
                  'freeze' if config.get('freeze_vit', False) else 'finetune',
                  args.run_id]
        )
        print("✅ W&B logging enabled")
    else:
        if wandb_api_key and not WANDB_AVAILABLE:
            print("⚠️  W&B API key provided but wandb not installed")
        print("ℹ️  W&B logging disabled")
    
    # Create model
    use_cross_piece = config.get('use_cross_piece_attention', False)
    print("\nCreating ViT-based model...")
    print(f"ViT Model: {config['vit_model_name']}")
    print(f"Freeze ViT: {config.get('freeze_vit', False)}")
    print(f"Freeze Layers: {config.get('freeze_layers', 0)}")
    print(f"Use Cross-Piece Attention: {use_cross_piece}")
    
    if use_cross_piece:
        print("✅ Using ViT V2 architecture (with cross-piece attention)")
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
    else:
        print("ℹ️  Using ViT V1 architecture (independent piece processing)")
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
    
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4),
        betas=config.get('betas', (0.9, 0.999))
    )
    
    # Setup scheduler
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        total_steps=total_steps,
        pct_start=config.get('warmup_pct', 0.1)
    )
    
    # Create flow matcher
    flow_matcher = DiscreteFlowMatcher(
        n_positions=actual_grid_size ** 2,
        interpolation_type=config.get('interpolation_type', 'linear')
    )
    
    # Determine validation batch size
    val_batch_size = args.val_batch_size if args.val_batch_size is not None else max(1, config['batch_size'] // 4)
    print(f"Validation batch size: {val_batch_size}")
    
    # Reconfigure validation loader with smaller batch size
    _, val_loader, _ = create_dataloaders(
        data_root=args.data_root,
        batch_size=val_batch_size,
        use_coordinates=False,
        num_workers=num_workers,
        normalize=True,
        pin_memory=config.get('pin_memory', True)
    )
    
    # Create trainer
    val_n_steps = config.get('val_n_steps', 20)
    print(f"Training n_steps: 1 (single step per training iteration)")
    print(f"Validation n_steps: {val_n_steps}")
    
    flow_trainer = FlowMatchingTrainer(
        model=model,
        flow_matcher=flow_matcher,
        optimizer=optimizer,
        device=device,
        use_amp=config.get('use_amp', False),
        val_n_steps=val_n_steps
    )
    
    if args.test_validation:
        print("\n" + "="*60)
        print("Running validation test to check for OOM errors...")
        print("="*60)
        
        try:
            model.eval()
            with torch.no_grad():
                pieces, target_positions = next(iter(val_loader))
                if target_positions.dim() == 3:
                    target_positions = target_positions[:, :, 0] * actual_grid_size + target_positions[:, :, 1]
                
                loss, metrics = flow_trainer.eval_step(pieces, target_positions)
                print(f"✅ Validation test passed!")
                print(f"   Loss: {metrics['loss']:.4f}")
                print(f"   Accuracy: {metrics['accuracy']:.4f}")
                print(f"   Sampling Accuracy: {metrics['sampling_accuracy']:.4f}")
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"❌ Validation OOM! Try --val-batch-size {val_batch_size // 2} or lower")
                raise
            else:
                raise
    
    # Create checkpoints directory
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to checkpoint directory
    config_save_path = checkpoint_dir / 'config.yaml'
    save_config(config, config_save_path)
    print(f"Saved config to: {config_save_path}")
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=0.0,
        mode='max'  # We want to maximize sampling accuracy
    )
    print(f"\nEarly stopping enabled with patience: {args.early_stopping_patience} epochs")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_samp_accuracy = 0.0
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_samp_accuracy = checkpoint.get('best_val_samp_accuracy', checkpoint.get('best_val_accuracy', 0.0))
        
        # Restore early stopping state if available
        if 'early_stopping_counter' in checkpoint:
            early_stopping.counter = checkpoint['early_stopping_counter']
            early_stopping.best_score = checkpoint.get('early_stopping_best_score', best_val_samp_accuracy)
            print(f"Restored early stopping state: counter={early_stopping.counter}, best_score={early_stopping.best_score:.4f}")
        
        print(f"Resuming from epoch {start_epoch}")
        print(f"Best validation sampling accuracy so far: {best_val_samp_accuracy:.4f}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    for epoch in range(start_epoch, config['num_epochs']):
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        # Training phase
        model.train()
        total_train_loss = 0.0
        total_train_accuracy = 0.0
        num_train_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Train')
        for pieces, target_positions in pbar:
            # Convert coordinates to indices if needed
            if target_positions.dim() == 3:
                target_positions = target_positions[:, :, 0] * actual_grid_size + target_positions[:, :, 1]
            
            # Training step
            loss, metrics = flow_trainer.train_step(pieces, target_positions)
            
            # Update scheduler
            scheduler.step()
            
            # Accumulate metrics
            total_train_loss += metrics['loss']
            total_train_accuracy += metrics['accuracy']
            num_train_batches += 1
            
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.3f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_train_loss = total_train_loss / num_train_batches
        avg_train_accuracy = total_train_accuracy / num_train_batches
        
        print(f"Train - Loss: {avg_train_loss:.4f}, Acc: {avg_train_accuracy:.4f}")
        
        # Log training metrics to W&B
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': avg_train_loss,
                'train/accuracy': avg_train_accuracy,
                'train/lr': optimizer.param_groups[0]['lr'],
                'early_stopping/counter': early_stopping.counter
            }, step=epoch)
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        total_val_accuracy = 0.0
        total_val_sampling_accuracy = 0.0
        num_val_batches = 0
        
        pbar = tqdm(val_loader, desc=f'Val')
        with torch.no_grad():
            for pieces, target_positions in pbar:
                # Convert coordinates to indices if needed
                if target_positions.dim() == 3:
                    target_positions = target_positions[:, :, 0] * actual_grid_size + target_positions[:, :, 1]
                
                # Evaluation step
                loss, metrics = flow_trainer.eval_step(pieces, target_positions)
                
                # Accumulate metrics
                total_val_loss += metrics['loss']
                total_val_accuracy += metrics['accuracy']
                total_val_sampling_accuracy += metrics['sampling_accuracy']
                num_val_batches += 1
                
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'acc': f"{metrics['accuracy']:.3f}",
                    'samp_acc': f"{metrics['sampling_accuracy']:.3f}"
                })
        
        avg_val_loss = total_val_loss / num_val_batches
        avg_val_accuracy = total_val_accuracy / num_val_batches
        avg_val_sampling_accuracy = total_val_sampling_accuracy / num_val_batches
        
        print(f"Val   - Loss: {avg_val_loss:.4f}, Acc: {avg_val_accuracy:.4f}, Samp Acc: {avg_val_sampling_accuracy:.4f}")
        
        # Log validation metrics to W&B
        if use_wandb:
            wandb.log({
                'val/loss': avg_val_loss,
                'val/accuracy': avg_val_accuracy,
                'val/sampling_accuracy': avg_val_sampling_accuracy,
            }, step=epoch)
        
        # Check if this is the best model
        is_best = avg_val_sampling_accuracy > best_val_samp_accuracy
        if is_best:
            best_val_samp_accuracy = avg_val_sampling_accuracy
            print(f"✅ New best model! (Samp Acc: {best_val_samp_accuracy:.4f})")
            
            # Log best model to W&B
            if use_wandb:
                wandb.run.summary['best_val_samp_accuracy'] = best_val_samp_accuracy
                wandb.run.summary['best_val_accuracy'] = avg_val_accuracy
                wandb.run.summary['best_epoch'] = epoch + 1
        
        # Save checkpoint
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_samp_accuracy': best_val_samp_accuracy,
            'best_val_accuracy': avg_val_accuracy,
            'config': config,
            'early_stopping_counter': early_stopping.counter,
            'early_stopping_best_score': early_stopping.best_score
        }
        
        # Save last checkpoint (for resumption)
        last_checkpoint_path = checkpoint_dir / 'last_checkpoint.pth'
        torch.save(checkpoint_data, last_checkpoint_path)
        
        # Save best model (only save best weights to save space)
        if is_best:
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint_data, best_path)
            print(f"💾 Saved best model to: {best_path}")
        
        # Check early stopping
        should_stop = early_stopping(avg_val_sampling_accuracy)
        if should_stop:
            print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
            print(f"   No improvement in validation sampling accuracy for {early_stopping.patience} epochs")
            print(f"   Best validation sampling accuracy: {best_val_samp_accuracy:.4f}")
            break
        
        if early_stopping.counter > 0:
            print(f"⏳ Early stopping counter: {early_stopping.counter}/{early_stopping.patience}")
        
        print()
    
    print("\n" + "="*60)
    print("🎉 Training complete!")
    print(f"Best validation sampling accuracy: {best_val_samp_accuracy:.4f}")
    if early_stopping.early_stop:
        print(f"Training stopped early at epoch {epoch + 1}")
    print("="*60)
    
    # Finish W&B run
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
