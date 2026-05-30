# Pomeranz et al. 2011 Greedy Jigsaw Puzzle Solver

Implementation of the greedy square jigsaw puzzle solver from:

**Dolev Pomeranz, Michal Shemesh, and Ohad Ben-Shahar.**  
*A fully automated greedy square jigsaw puzzle solver.*  
In CVPR 2011, pages 9–16. IEEE, 2011.

## Overview

This is a classical (non-learning based) jigsaw puzzle solver that uses:
- **Compatibility metric**: Based on edge color dissimilarity in LAB color space
- **Best buddies**: Mutual nearest neighbors based on compatibility
- **Greedy placement**: Iteratively places pieces that fit best with already placed pieces
- **Segmentation**: Removes incorrectly placed pieces and refines the solution

The implementation is adapted from [yi-jiayu/shuffled-images](https://github.com/yi-jiayu/shuffled-images) to work with the GAP dataset format (HDF5).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run solver on test set

```bash
# GAP-3 test set
python run_solver.py \
    --data_dir path/to/GAP-3/test \
    --output results/gap3_predictions.npy \
    --num_trials 10

# GAP-5 test set
python run_solver.py \
    --data_dir path/to/GAP-5/test \
    --output results/gap5_predictions.npy \
    --num_trials 10
```

**Arguments:**
- `--data_dir`: Path to dataset directory containing `puzzles.h5` and `labels_indices.h5`
- `--output`: Path to save predictions (`.npy` file)
- `--num_trials`: Number of random seed trials per puzzle (default: 10)
- `--max_samples`: Optional limit on number of samples for testing
- `--seed`: Random seed for reproducibility

### Evaluate predictions

```bash
# Evaluate GAP-3
python evaluate.py \
    --predictions results/gap3_predictions.npy \
    --data_dir path/to/GAP-3/test

# Evaluate GAP-5
python evaluate.py \
    --predictions results/gap5_predictions.npy \
    --data_dir path/to/GAP-5/test
```

**Metrics computed:**
- **Perfect Rate**: Fraction of puzzles perfectly reconstructed
- **Piece Accuracy**: Fraction of pieces in correct positions
- **Neighbor Accuracy**: Fraction of adjacent piece pairs both in correct positions

### Quick test on small subset

```bash
# Test on first 10 puzzles
python run_solver.py \
    --data_dir path/to/GAP-3/test \
    --output results/test_predictions.npy \
    --num_trials 5 \
    --max_samples 10

python evaluate.py \
    --predictions results/test_predictions.npy \
    --data_dir path/to/GAP-3/test
```

## Algorithm Details

### Compatibility Metric

For each pair of pieces and each edge direction, computes:

$$D(x_i, x_j) = \sum \left[ |G_L(x_i) - x_j^L|^p + |G_L(x_j) - x_i^R|^p \right]^{q/p}$$

where:
- $G_L(x_i)$ is the left gradient boundary of piece $x_i$
- $p = 0.3, q = 0.0625$ (as in original paper)
- Images are in LAB color space

Compatibility is then: $C(x_i, x_j) = \exp(-D(x_i, x_j) / \tau)$

### Best Buddies

Two pieces are "best buddies" if they are mutually the best match for each other.

### Greedy Placement

1. Start with a seed piece (one with most best buddies)
2. Find candidate slots adjacent to placed pieces
3. For each slot, try pieces that are best buddies with all neighbors
4. Place piece with highest average compatibility
5. Repeat until all pieces placed

### Refinement

After initial placement:
1. Segment puzzle into connected components (based on best buddies)
2. Keep only largest segment
3. Re-center and place remaining pieces
4. Repeat until score stops improving

## Output Format

- **Predictions**: `.npy` file with shape `(N, num_pieces)` containing piece indices
- **Info**: `_info.npz` file with:
  - `scores`: Best buddies scores for each puzzle
  - `times`: Solving time for each puzzle
  - `num_trials`: Number of trials used

## Performance Notes

- **Runtime**: ~5-30 seconds per puzzle (depending on size and number of trials)
- **Multiple trials**: Running with multiple random seeds typically improves results
- **Grid size**: Works for square puzzles (3x3, 5x5, etc.)
- **Color space**: Uses LAB color space as recommended in the paper

## References

```bibtex
@inproceedings{pomeranz2011fully,
  title={A fully automated greedy square jigsaw puzzle solver},
  author={Pomeranz, Dolev and Shemesh, Michal and Ben-Shahar, Ohad},
  booktitle={CVPR 2011},
  pages={9--16},
  year={2011},
  organization={IEEE}
}
```

Project page: http://www.cs.bgu.ac.il/~icvl/projects/project-jigsaw.html
