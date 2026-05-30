# Sholomon et al. 2013 Genetic Algorithm Jigsaw Puzzle Solver

Implementation of the genetic algorithm-based jigsaw puzzle solver from:

**Dror Sholomon, Omid David, and Nathan S Netanyahu.**  
*A genetic algorithm-based solver for very large jigsaw puzzles.*  
In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1767–1774, 2013.

## Overview

This is a classical (non-learning based) jigsaw puzzle solver that uses genetic algorithms with:
- **Fitness function**: Based on edge color dissimilarity (same as Pomeranz et al.)
- **Chromosome representation**: Permutation of piece indices
- **Crossover**: Partially Mapped Crossover (PMX) for permutations
- **Mutation**: Swap, inversion, and scramble mutations
- **Selection**: Tournament selection with elitism
- **Evolution**: Iterative improvement over multiple generations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run solver on test set

```bash
# GAP-3 test set
python run_solver.py \
    --data_dir /path/to/GAP-3/test \
    --output results/gap3_predictions.npy \
    --pop_size 100 \
    --n_generations 1000 \
    --mutation_rate 0.01

# GAP-5 test set
python run_solver.py \
    --data_dir /path/to/GAP-5/test \
    --output results/gap5_predictions.npy \
    --pop_size 150 \
    --n_generations 2000 \
    --mutation_rate 0.01
```

**Arguments:**
- `--data_dir`: Path to dataset directory containing `puzzles.h5` and `labels_indices.h5`
- `--output`: Path to save predictions (`.npy` file)
- `--pop_size`: Population size (default: 100; larger for harder puzzles)
- `--n_generations`: Maximum generations (default: 1000; more for harder puzzles)
- `--mutation_rate`: Mutation probability (default: 0.01)
- `--max_samples`: Optional limit on number of samples for testing
- `--seed`: Random seed for reproducibility

### Evaluate predictions

```bash
# Evaluate GAP-3
python evaluate.py \
    --predictions results/gap3_predictions.npy \
    --data_dir /path/to/GAP-3/test

# Evaluate GAP-5
python evaluate.py \
    --predictions results/gap5_predictions.npy \
    --data_dir /path/to/GAP-5/test
```

**Metrics computed:**
- **Perfect Rate**: Fraction of puzzles perfectly reconstructed
- **Piece Accuracy**: Fraction of pieces in correct positions
- **Neighbor Accuracy**: Fraction of adjacent piece pairs both in correct positions

### Quick test on small subset

```bash
# Test on first 5 puzzles with smaller parameters
python run_solver.py \
    --data_dir /path/to/GAP-3/test \
    --output results/test_predictions.npy \
    --pop_size 50 \
    --n_generations 500 \
    --max_samples 5

python evaluate.py \
    --predictions results/test_predictions.npy \
    --data_dir /path/to/GAP-3/test
```

## Algorithm Details

### Genetic Algorithm Components

#### 1. Fitness Function

Fitness is the negative sum of edge dissimilarities between adjacent pieces:

$$\text{Fitness} = -\sum_{(i,j) \text{ adjacent}} D(piece_i, piece_j)$$

where $D$ is the edge dissimilarity metric (same as Pomeranz et al.).

#### 2. Initialization

- Random permutations of piece indices
- Population size: typically 100-200 individuals

#### 3. Selection

- **Tournament Selection**: Select best from random subset
- **Elitism**: Keep top 10% of individuals across generations

#### 4. Crossover (PMX)

Partially Mapped Crossover for permutations:
- Select two crossover points
- Exchange middle sections between parents
- Fix conflicts to maintain valid permutations

#### 5. Mutation

Three mutation operators applied randomly:
- **Swap**: Exchange two random pieces
- **Inversion**: Reverse a random subsequence
- **Scramble**: Shuffle a random subsequence

#### 6. Evolution

- Run for up to `n_generations` iterations
- Early stopping if no improvement for 100 generations
- Track best solution across all generations

### Parameter Recommendations

**For GAP-3 (3x3 = 9 pieces):**
- Population size: 50-100
- Generations: 500-1000
- Mutation rate: 0.01-0.02

**For GAP-5 (5x5 = 25 pieces):**
- Population size: 100-200
- Generations: 1000-2000
- Mutation rate: 0.01-0.02

**For larger puzzles:**
- Increase population size (200-500)
- Increase generations (2000-5000)
- May need lower mutation rate (0.005-0.01)

## Output Format

- **Predictions**: `.npy` file with shape `(N, num_pieces)` containing piece indices
- **Info**: `_info.npz` file with:
  - `fitness_scores`: Fitness values for each puzzle
  - `generations_used`: Number of generations until convergence
  - `times`: Solving time for each puzzle (seconds)
  - Algorithm parameters (pop_size, n_generations, mutation_rate)

## Performance Notes

- **Runtime**: ~10-60 seconds per puzzle depending on parameters and puzzle size
- **Quality**: Generally produces good solutions but may not always find global optimum
- **Scalability**: Works well for puzzles up to ~10x10 (100 pieces)
- **Randomness**: Different runs may produce different results (use `--seed` for reproducibility)

### Expected Runtimes

- **GAP-3** (pop=100, gen=1000): ~20-30 seconds per puzzle
- **GAP-5** (pop=150, gen=2000): ~60-120 seconds per puzzle
- Full test set (3000 puzzles): Several hours

## Comparison with Greedy Solver

**Genetic Algorithm Advantages:**
- Can escape local optima through mutation and crossover
- Population-based search explores multiple solutions simultaneously
- Often finds better solutions for larger puzzles

**Genetic Algorithm Disadvantages:**
- Slower than greedy solver (especially for small puzzles)
- Non-deterministic results (even with same seed, order matters)
- Requires parameter tuning

**Greedy Solver Advantages:**
- Faster, especially for small puzzles
- Deterministic given seed piece
- No parameter tuning needed

**Greedy Solver Disadvantages:**
- Can get stuck in local optima
- Quality degrades for larger puzzles

## References

```bibtex
@inproceedings{sholomon2013genetic,
  title={A genetic algorithm-based solver for very large jigsaw puzzles},
  author={Sholomon, Dror and David, Omid Elyasaf and Netanyahu, Nathan S},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1767--1774},
  year={2013}
}
```

## Implementation Notes

This implementation includes:
- PMX (Partially Mapped Crossover) for permutation chromosomes
- Multiple mutation operators (swap, inversion, scramble)
- Tournament selection with elitism
- Early stopping for efficiency
- Same edge compatibility metric as Pomeranz et al. for fair comparison
