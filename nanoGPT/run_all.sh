#!/usr/bin/env bash
# run_all.sh -- sequential GPT training runs (Uni-mHC + baselines), restart-safe.
# Each run auto-resumes from its newest checkpoint, so this script can be
# re-launched after any interruption and it will continue where it stopped.
cd "$(dirname "$0")"
PY=~/venv-torch/bin/python
mkdir -p ../runs

run_hc () {  # name mixer extra...
  local name=$1; local mixer=$2; shift 2
  echo "=== [$(date +%H:%M:%S)] RUN $name (mixer=$mixer) ==="
  $PY train_hc.py --mixer="$mixer" --out_dir="../runs/$name" "$@" 2>&1 | tail -4
}

run_hc unimhc   unimhc   --max_iters=1000
run_hc givens   givens   --max_iters=1000
run_hc sinkhorn sinkhorn --max_iters=1000
run_hc ortho    ortho    --max_iters=1000

echo "=== [$(date +%H:%M:%S)] RUN vanilla (nanoGPT reference, no HC) ==="
$PY train.py --dataset=shakespeare_char --out_dir=../runs/vanilla \
  --n_layer=6 --n_head=6 --n_embd=384 --block_size=256 --batch_size=32 \
  --max_iters=1000 --eval_interval=100 --eval_iters=50 --lr=1e-3 \
  --warmup_iters=100 --lr_decay_iters=1000 2>&1 | tail -4

# generate a sample from the trained Uni-mHC model for the report
$PY sample_hc.py --ckpt=../runs/unimhc/ckpt_final.pt --out=../runs/unimhc/sample.txt 2>&1 | tail -2
echo "=== [$(date +%H:%M:%S)] ALL RUNS DONE ==="
