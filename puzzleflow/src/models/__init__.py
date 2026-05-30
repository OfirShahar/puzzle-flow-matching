"""
Models module for puzzle reassembly with flow matching.
"""

from .unet import PuzzleFlowTransformer, PieceEncoder, TransformerBlock
from .flow_matcher import DiscreteFlowMatcher, FlowMatchingTrainer
from .vit_flow import PuzzleFlowViT, create_vit_model
from .vit_flow_v2 import PuzzleFlowViT as PuzzleFlowViT_V2, create_vit_model as create_vit_model_v2
from .cnn_flow import PuzzleFlowCNN, create_cnn_model
from .vit_flow_ablations import (
    PuzzleFlowViT_DirectPrediction,
    PuzzleFlowViT_CustomSchedule,
    PuzzleFlowViT_VariableLayers,
    create_direct_prediction_model,
    create_schedule_ablation_model,
    create_layers_ablation_model,
)

__all__ = [
    'PuzzleFlowTransformer',
    'PieceEncoder',
    'TransformerBlock',
    'DiscreteFlowMatcher',
    'FlowMatchingTrainer',
    'PuzzleFlowViT',
    'create_vit_model',
    'PuzzleFlowViT_V2',
    'create_vit_model_v2',
    'PuzzleFlowCNN',
    'create_cnn_model',
    # Ablation models
    'PuzzleFlowViT_DirectPrediction',
    'PuzzleFlowViT_CustomSchedule',
    'PuzzleFlowViT_VariableLayers',
    'create_direct_prediction_model',
    'create_schedule_ablation_model',
    'create_layers_ablation_model',
]
