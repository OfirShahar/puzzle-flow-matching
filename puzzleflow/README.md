# PuzzleFlow

**PuzzleFlow** is a Vision Transformer and discrete Flow Matching-based framework for solving jigsaw puzzles with irregular, archaeologically-inspired fragment shapes. It was introduced alongside the GAP (Generated Archaeological-fragments Puzzles) benchmark datasets.

## Architecture Overview

PuzzleFlow consists of three main components:

1. **ViT Feature Encoder** (`src/models/vit_flow.py`): A pretrained ViT-Base/16 (`google/vit-base-patch16-224`) extracts per-piece visual features. A learned 1×1 convolution adapts RGBA inputs (4 channels, including the alpha mask encoding fragment shape) to the ViT's expected 3-channel input. The ViT backbone is fine-tuned during training to learn features specific to eroded, irregular fragments.

2. **Discrete Flow Matcher** (`src/models/flow_matcher.py`): Implements discrete flow matching over permutations. During training, it interpolates between a random permutation (noise at $t{=}0$) and the ground-truth permutation ($t{=}1$), training the model to predict the velocity field on this discrete interpolation path. At inference, it iteratively refines a random initial permutation through $S$ denoising steps (default: 20), using greedy assignment at each step.

3. **Task-Specific Transformer Layers** (`src/models/vit_flow.py`): After ViT feature extraction, $L{=}4$ additional transformer encoder layers (with 12 heads and 1024-dim feedforward) perform cross-piece relational reasoning, enabling the model to capture global spatial coherence across all fragments simultaneously.

### Pipeline

```
Input: Shuffled puzzle pieces (N, C, H, W) with RGBA channels
  → ViT Feature Encoder (per-piece)
    → Task-Specific Transformer Layers (cross-piece attention)
      → Discrete Flow Matching (iterative permutation refinement)
        → Output: Predicted permutation (piece → position mapping)
```

## Directory Structure

```
puzzleflow/
├── configs/
│   └── vit_config.yaml          # Default training configuration
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── predict.py               # Prediction / inference
│   └── evaluate.py              # Evaluation metrics
└── src/
    ├── models/
    │   ├── vit_flow.py          # PuzzleFlowViT model
    │   └── flow_matcher.py      # Discrete flow matching
    ├── data/
    │   └── puzzle_dataset.py    # HDF5 dataset loader
    ├── training/
    │   ├── trainer.py           # Training loop
    │   ├── losses.py            # Loss functions
    │   └── metrics.py           # Training metrics
    └── utils/
        ├── config.py            # YAML config loading
        ├── checkpoint.py        # Checkpoint save/load
        └── visualization.py     # Puzzle visualization
```

## Requirements

```
torch >= 2.0
transformers >= 4.35  (for ViT model)
timm
h5py
numpy
pyyaml
tqdm
matplotlib
```

## Usage

### Training

```bash
python scripts/train.py \
    --config configs/vit_config.yaml \
    --data-root /path/to/GAP-3 \
    --checkpoint-dir checkpoints/vit_GAP-3 \
    --run-id run1 \
    --epochs 30 \
    --batch-size 8 \
    --num-workers 8 \
    --use-amp
```

**Key training arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/vit_config.yaml` | Path to YAML configuration |
| `--data-root` | (required) | Path to dataset directory (e.g., `data/GAP-3`) |
| `--checkpoint-dir` | from config | Directory for saving checkpoints |
| `--run-id` | `run1` | Identifier for this training run |
| `--epochs` | from config (30) | Number of training epochs |
| `--batch-size` | from config (8) | Training batch size |
| `--resume` | None | Path to checkpoint for resuming training |
| `--freeze` | False | Freeze ViT backbone (feature extraction only) |
| `--use-amp` | False | Enable mixed-precision training |
| `--early-stopping-patience` | 10 | Epochs to wait before early stopping |

### Prediction

```bash
python scripts/predict.py \
    --checkpoint checkpoints/vit_GAP-3/run1/best_model.pth \
    --data-root /path/to/GAP-3 \
    --split test \
    --output predictions/puzzleflow_GAP-3_predictions.npz \
    --n-steps 20 \
    --batch-size 4
```

**Key prediction arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Path to trained model checkpoint |
| `--data-root` | (required) | Path to dataset directory |
| `--split` | `test` | Dataset split to evaluate (`train`, `val`, `test`) |
| `--output` | (required) | Output path for predictions (`.npz`) |
| `--n-steps` | 20 | Number of flow matching inference steps |
| `--batch-size` | 4 | Inference batch size |

The prediction script outputs an `.npz` file containing:
- `predictions`: `(N, num_pieces)` — predicted position for each piece
- `targets`: `(N, num_pieces)` — ground-truth positions
- Metadata (checkpoint path, dataset, split, number of steps)

### Evaluation

```bash
python scripts/evaluate.py \
    --predictions predictions/puzzleflow_GAP-3_predictions.npz
```

This computes:
- **PA** (Perfect Accuracy): % of fully solved puzzles
- **AA** (Absolute Accuracy): % of correctly placed pieces
- **SRA** (Spatial Relationship Accuracy): % of preserved neighbor relationships

## Configuration

The default configuration (`configs/vit_config.yaml`) specifies:

```yaml
vit_model_name: 'google/vit-base-patch16-224'
freeze_vit: false
piece_size: [96, 96]
grid_size: 3          # Auto-detected from dataset
d_model: 768
n_heads: 12
n_layers: 4           # Task-specific transformer layers
dim_feedforward: 3072
dropout: 0.1
batch_size: 8
num_epochs: 30
learning_rate: 0.00001
weight_decay: 0.01
```

For GAP-5 (5×5 puzzles), increase `num_epochs` to 50 and consider reducing `batch_size` to 4 if memory is limited.

## Dataset Format

PuzzleFlow expects datasets in HDF5 format with the following structure:

```
dataset_root/
├── train/
│   ├── puzzles.h5          # (N, n_pieces, H, W, C) uint8
│   ├── labels_indices.h5   # (N, n_pieces) int — permutation indices
│   ├── labels_coordinates.h5  # (N, n_pieces, 2) int — (row, col)
│   └── metadata.json
├── val/
│   └── ...
└── test/
    └── ...
```

The model uses `labels_indices.h5` (1D permutation format) for training and evaluation.

## Citation

If you use PuzzleFlow or the GAP datasets in your research, please cite:

```bibtex
@article{shahar2026missing,
  title={The Missing GAP: From Solving Square Jigsaw Puzzles to Handling Real World Archaeological Fragments},
  author={Shahar, Ofir Itzhak and Elkin, Gur and Ben-Shahar, Ohad},
  journal={arXiv preprint arXiv:2605.12077},
  year={2026}
}
```

> The citation above will be replaced with the official CVPR 2026 reference once available.
