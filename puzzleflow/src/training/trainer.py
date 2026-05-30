"""
Main training script for puzzle reassembly with flow matching.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import os
import sys
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Optional, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models import PuzzleFlowTransformer, DiscreteFlowMatcher, FlowMatchingTrainer
from src.training.metrics import compute_puzzle_metrics
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.config import load_config


class Trainer:
    """
    Complete trainer for puzzle flow matching.
    """
    def __init__(
        self,
        model: nn.Module,
        flow_matcher: DiscreteFlowMatcher,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: Optional[torch.device] = None,
        use_wandb: bool = False
    ):
        self.model = model
        self.flow_matcher = flow_matcher
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_wandb = use_wandb
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4),
            betas=config.get('betas', (0.9, 0.999))
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * config['num_epochs']
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            total_steps=total_steps,
            pct_start=config.get('warmup_pct', 0.1)
        )
        
        # Create flow matching trainer
        self.flow_trainer = FlowMatchingTrainer(
            model=self.model,
            flow_matcher=self.flow_matcher,
            optimizer=self.optimizer,
            device=self.device
        )
        
        # Tracking
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if requested
        if self.use_wandb:
            # Check for W&B API key in config or environment
            wandb_api_key = config.get('wandb_api_key')
            if wandb_api_key:
                os.environ['WANDB_API_KEY'] = wandb_api_key
            
            wandb.init(
                project=config.get('project_name', 'puzzle-flow-matching'),
                config=config,
                name=config.get('run_name', None)
            )
            print("✅ W&B logging enabled")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch_idx, (pieces, target_positions) in enumerate(pbar):
            # Convert coordinates to indices if needed
            if target_positions.dim() == 3:  # (B, N, 2) format
                grid_size = self.model.grid_size
                target_positions = target_positions[:, :, 0] * grid_size + target_positions[:, :, 1]
            
            # Training step
            loss, metrics = self.flow_trainer.train_step(pieces, target_positions)
            
            # Update scheduler
            self.scheduler.step()
            
            # Accumulate metrics
            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.3f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': metrics['loss'],
                    'train/accuracy': metrics['accuracy'],
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/step': self.current_epoch * len(self.train_loader) + batch_idx
                })
        
        # Epoch metrics
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_sampling_accuracy = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')
        
        for pieces, target_positions in pbar:
            # Convert coordinates to indices if needed
            if target_positions.dim() == 3:  # (B, N, 2) format
                grid_size = self.model.grid_size
                target_positions = target_positions[:, :, 0] * grid_size + target_positions[:, :, 1]
            
            # Evaluation step
            loss, metrics = self.flow_trainer.eval_step(pieces, target_positions)
            
            # Accumulate metrics
            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            total_sampling_accuracy += metrics['sampling_accuracy']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.3f}",
                'samp_acc': f"{metrics['sampling_accuracy']:.3f}"
            })
        
        # Epoch metrics
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_sampling_accuracy = total_sampling_accuracy / num_batches
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'sampling_accuracy': avg_sampling_accuracy
        }
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            print(f"\nEpoch {epoch + 1} Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Accuracy: {train_metrics['accuracy']:.3f}")
            
            # Validate
            val_metrics = self.validate()
            print(f"Epoch {epoch + 1} Val - Loss: {val_metrics['loss']:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.3f}, "
                  f"Sampling Accuracy: {val_metrics['sampling_accuracy']:.3f}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_metrics['loss'],
                    'train/epoch_accuracy': train_metrics['accuracy'],
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/sampling_accuracy': val_metrics['sampling_accuracy']
                })
            
            # Save checkpoint (using accuracy instead of sampling_accuracy)
            is_best = val_metrics['accuracy'] > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_metrics['accuracy']
                
                # Log best model to W&B
                if self.use_wandb:
                    wandb.run.summary['best_val_accuracy'] = self.best_val_accuracy
                    wandb.run.summary['best_epoch'] = epoch + 1
            
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_accuracy': self.best_val_accuracy,
                    'config': self.config
                },
                is_best=is_best,
                checkpoint_dir=self.checkpoint_dir,
                filename=f'checkpoint_epoch_{epoch + 1}.pt'
            )
            
            print(f"{'✅ New best model!' if is_best else ''}\n")
        
        print(f"\n🎉 Training complete! Best validation accuracy: {self.best_val_accuracy:.3f}")
        
        if self.use_wandb:
            wandb.finish()


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train puzzle flow matching model')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--wandb', action='store_true',
                        help='Use Weights & Biases logging')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Import dataset
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from the_missing_gap.puzzle_dataset import create_dataloaders
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=config['batch_size'],
        use_coordinates=False,  # Use index format
        num_workers=config.get('num_workers', 4),
        normalize=True
    )
    
    # Create model and flow matcher
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PuzzleFlowTransformer(
        piece_size=config['piece_size'],
        grid_size=config['grid_size'],
        in_channels=3,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    )
    
    flow_matcher = DiscreteFlowMatcher(
        n_positions=config['grid_size'] ** 2,
        interpolation_type=config.get('interpolation_type', 'linear')
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        flow_matcher=flow_matcher,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        use_wandb=args.wandb
    )
    
    # Train
    trainer.train()
