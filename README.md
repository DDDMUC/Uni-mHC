# Uni-mHC

PyTorch implementation of Uni-mHC (Unistochastic Manifold-Constrained
Hyper-Connections) and its integration into nanoGPT.

mHC (DeepSeek, 2025) constrains the stream-mixing matrix of hyper-connections
to the Birkhoff polytope of doubly stochastic matrices via Sinkhorn-Knopp
iteration. That buys energy conservation, but the enforced non-negativity makes
mixing purely additive: a stream can never cancel another stream's features.
Uni-mHC replaces the iterative projection with an exact complex Cayley
transform — a unitary U whose |U|^2 is doubly stochastic by construction — and
reads out signed weights through phase interference, so it keeps the energy
guarantee and regains subtraction.

## What's here

- `models/manifold_layers.py` — four drop-in mixing operators:
  `UniMHCResidual` (ours, complex Cayley transform), `GivensMHCResidual`
  (ours-fast, inverse-free Givens rotations), `SinkhornMHCResidual`
  (DeepSeek mHC baseline, 20 SK iterations), `OrthostochasticMHCResidual`
  (Newton-Schulz polar decomposition baseline).
- `fast_exp.py` — operator benchmarks and the subtractive-disentanglement
  toy task.
- `nanoGPT/` — 4-stream char-GPT (`model_hc.py`) and a pausable trainer
  (`train_hc.py`).
- `paper/` — the 4-page preprint; its tables are generated from `results/`
  and `runs/` by `paper/make_macros.py`, so the PDF cannot drift from the
  logged numbers.
- `runs/` — logs and metrics of five 1000-step training runs (weights not
  included).

## Quick start

```bash
pip install torch numpy matplotlib   # CUDA wheel: see pytorch.org
python fast_exp.py                   # operator bench + toy task, ~3 min on an RTX 4060
```

Train a 4-stream char-GPT (~6 min per mixer on an RTX 4060):

```bash
cd nanoGPT
python data/shakespeare_char/prepare.py
python train_hc.py --mixer=unimhc --out_dir=../runs/unimhc
```

Training checkpoints every 200 steps; rerun the same command after an
interruption and it resumes from the newest checkpoint.

## Results

All runs on an RTX 4060 Laptop (8 GB), float32, stream width n=4.

| Operator | row/col-sum error | ms/iter | toy loss (Y=X0-X1) |
|---|---|---|---|
| UniMHC (ours) | ~5e-7, exact by construction | 10.6 | 1.1e-11, coeffs (+1,-1,0,0) |
| GivensMHC (ours-fast) | ~4e-7 | 10.9 | 9.7e-15, coeffs (+1,-1,0,0) |
| Sinkhorn mHC | col-sum 5e-3 after 20 iters | 12.9 | 0.986 (floor 1.0) |
| Orthostochastic | ~2e-7 | 11.2 | 0.986 (floor 1.0) |

nanoGPT, 6 layers / 384 dims / 4 streams, 1000 steps on shakespeare_char: all
constrained mixers train stably and reach val loss 1.505-1.523. An identically
trained vanilla GPT overfits inside the short horizon (train 0.112, val 3.459).

## Honest status

- The subtraction result is clean: Uni-mHC solves the toy task to machine
  precision, and both non-negative baselines stop exactly at the provable
  floor of 0.5 Var(Y).
- On the language model, all four constrained mixers tie within noise at this
  scale. Mixers start at identity, and 1000 steps is too short for the extra
  freedom to matter. The overfitting resistance comes from the
  identity-initialized multi-stream topology itself, not from unistochastic
  mixing.
- Whether the subtractive freedom helps language modeling at depth is untested.
  That is the first experiment we would run next (deeper stacks, longer
  training); see the future-work list in the paper.

## Publication

Paper: `paper/Uni-mHC_Zenodo_v2.pdf` (author version; `v1` is an earlier
anonymous draft, kept for provenance).

Author: DDDMUC (暮迟), Independent Researcher.
License: MIT for this code; `nanoGPT/` contains Apache-2.0 code from
karpathy/nanoGPT (see `nanoGPT/LICENSE`).
