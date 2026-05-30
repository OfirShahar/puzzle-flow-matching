"""
Configuration management utilities.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        config: Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        if save_path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False)
        elif save_path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {save_path.suffix}")
    
    print(f"Config saved to: {save_path}")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for puzzle flow matching.
    
    Returns:
        config: Default configuration dictionary
    """
    return {
        # Model
        'piece_size': (64, 64),
        'grid_size': 4,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        
        # Flow matching
        'interpolation_type': 'linear',
        
        # Training
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999),
        'warmup_pct': 0.1,
        
        # Data
        'num_workers': 4,
        'pin_memory': True,
        
        # Checkpointing
        'checkpoint_dir': './checkpoints',
        'save_frequency': 1,
        
        # Logging
        'project_name': 'puzzle-flow-matching',
        'run_name': None,
        'log_frequency': 10,
    }
