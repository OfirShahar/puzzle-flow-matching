# Classical Baselines

Reimplementations of two classical (non-learning) jigsaw puzzle solvers, adapted to the [GAP](https://huggingface.co/datasets/Ofirish/GAP) dataset format. Both are reported as baselines in:

> **The Missing GAP: From Solving Square Jigsaw Puzzles to Handling Real World Archaeological Fragments.** Shahar, Elkin, and Ben-Shahar. CVPR 2026.

## Solvers

| Folder | Method | Original Paper |
|---|---|---|
| [`greedy-solver/`](greedy-solver) | Greedy best-buddies solver | Pomeranz, Shemesh, and Ben-Shahar. *A fully automated greedy square jigsaw puzzle solver.* CVPR 2011. |
| [`genetic-solver/`](genetic-solver) | Genetic algorithm with PMX crossover | Sholomon, David, and Netanyahu. *A genetic algorithm-based solver for very large jigsaw puzzles.* CVPR 2013. |

Each solver has its own `README.md`, `requirements.txt`, `run_solver.py`, and `evaluate.py`. Both consume the standard GAP HDF5 layout (`puzzles.h5` + `labels_indices.h5`) and emit `.npy` predictions compatible with the GAP evaluation script.

## Quick example (greedy)

```bash
cd greedy-solver
pip install -r requirements.txt
python run_solver.py --data_dir path/to/GAP-3/test --output results/gap3.npy --num_trials 10
python evaluate.py   --predictions results/gap3.npy --data_dir path/to/GAP-3/test
```

## Quick example (genetic)

```bash
cd genetic-solver
pip install -r requirements.txt
python run_solver.py --data_dir path/to/GAP-3/test --output results/gap3.npy \
    --pop_size 100 --n_generations 1000
python evaluate.py   --predictions results/gap3.npy --data_dir path/to/GAP-3/test
```

See the per-solver READMEs for full argument lists, parameter recommendations, and algorithm details.

## Citation

If you use these baselines, please cite both the original works (above) and the GAP paper:

```bibtex
@article{shahar2026missing,
  title={The Missing GAP: From Solving Square Jigsaw Puzzles to Handling Real World Archaeological Fragments},
  author={Shahar, Ofir Itzhak and Elkin, Gur and Ben-Shahar, Ohad},
  journal={arXiv preprint arXiv:2605.12077},
  year={2026}
}
```
