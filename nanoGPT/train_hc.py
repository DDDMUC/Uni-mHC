"""
train_hc.py -- pausable nanoGPT training for Uni-mHC / manifold mixers.

Adapted from karpathy/nanoGPT (train.py). Differences:
  * builds GPT from model_hc.py (n_streams residual streams + manifold mixer)
  * --mixer {unimhc,givens,sinkhorn,ortho}, --n_streams (default 4)
  * pausable: saves ckpt_latest.pt every --ckpt_every iters, auto-resumes from the
    newest checkpoint in --out_dir when --init_from=resume (default), appends eval
    metrics to metrics.jsonl (restart-safe), and saves ckpt_final.pt at the end.
Run (shakespeare_char):
  python train_hc.py --mixer=unimhc --out_dir=../runs/unimhc
Resume after interruption (same command): it picks up where it left off.
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from model_hc import GPT, GPTConfig

# -----------------------------------------------------------------------------


def get_batch(split, train_data, val_data, block_size, batch_size, device):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(
        (data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(
        (data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, ctx, eval_iters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, *ctx)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def find_latest_ckpt(out_dir: Path):
    cands = list(out_dir.glob("ckpt_latest.pt")) + list(out_dir.glob("ckpt_iter*.pt")) + list(out_dir.glob("ckpt_final.pt"))
    if not cands:
        return None
    def itern(p):
        s = p.stem
        if "latest" in s or "final" in s:
            try:
                meta = json.loads((p.parent / (s.replace("ckpt_", "meta_") + ".json")).read_text())
                return meta.get("iter", 0)
            except Exception:
                return 0
        return int(s.split("iter")[1])
    return max(cands, key=itern)


def save_ckpt(model, optimizer, iter_num, best_val, out_dir: Path, name):
    raw = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
           "model_args": vars(model.config), "iter_num": iter_num, "best_val_loss": best_val,
           "config": vars(g_args)}
    torch.save(raw, out_dir / f"{name}.pt")
    (out_dir / f"meta_{name}.json").write_text(json.dumps({"iter": iter_num, "best_val_loss": best_val}))


# -----------------------------------------------------------------------------


g_args = None
ctx_device_type = "cpu"


def main():
    global g_args, ctx_device_type
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=Path("../runs/unimhc"))
    p.add_argument("--init_from", default="resume", choices=["scratch", "resume"])
    # manifold mixing
    p.add_argument("--mixer", default="unimhc", choices=["unimhc", "givens", "sinkhorn", "ortho"])
    p.add_argument("--n_streams", type=int, default=4)
    # data / model (nanoGPT shakespeare_char defaults)
    p.add_argument("--dataset", type=Path, default=Path("data/shakespeare_char"))
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--bias", action="store_true")
    # train
    p.add_argument("--max_iters", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--decay_lr", action="store_true", default=True)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--lr_decay_iters", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=200)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    g_args = p.parse_args()
    args = g_args

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train_log.txt", "a", encoding="utf-8")

    def log(msg):
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx_device_type = device
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_dir = Path(__file__).resolve().parent / args.dataset
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]

    model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      block_size=args.block_size, bias=args.bias, vocab_size=vocab_size,
                      dropout=args.dropout, n_streams=args.n_streams, mixer=args.mixer)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(device)

    # resume bookkeeping ------------------------------------------------------
    iter_num = 0
    best_val_loss = 1e9
    optimizer = None
    if args.init_from == "resume":
        ck = find_latest_ckpt(out_dir)
        if ck is not None:
            state = torch.load(ck, map_location=device, weights_only=False)
            iter_num = state["iter_num"]
            best_val_loss = state["best_val_loss"]
            model.load_state_dict(state["model"])
            log(f"resumed from {ck.name} at iter {iter_num} (best_val {best_val_loss:.4f})")
    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device)
    if args.init_from == "resume" and iter_num > 0:
        optimizer.load_state_dict(state["optimizer"])

    if args.compile and sys.platform == "linux":
        log("using torch.compile")
        model = torch.compile(model)

    metrics_f = open(out_dir / "metrics.jsonl", "a", encoding="utf-8")
    ctx = (train_data, val_data, args.block_size, args.batch_size, device)

    def get_lr(it):
        if not args.decay_lr:
            return args.learning_rate
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / (args.warmup_iters + 1)
        if it > args.lr_decay_iters:
            return args.min_lr
        ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
        return args.min_lr + (args.learning_rate - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * ratio))

    X, Y = get_batch("train", *ctx)
    t0 = time.time()
    local_iter = 0
    model.train()
    log(f"start: mixer={args.mixer} n_streams={args.n_streams} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
        f"device={device} resume_iter={iter_num} max_iters={args.max_iters}")

    while True:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1:
            losses = estimate_loss(model, ctx, args.eval_iters)
            log(f"iter {iter_num:5d}: train {losses['train']:.4f}, val {losses['val']:.4f}, lr {lr:.2e}")
            metrics_f.write(json.dumps({"iter": iter_num, **losses, "lr": lr,
                                        "mixer": args.mixer, "elapsed_s": round(time.time() - t0, 1)}) + "\n")
            metrics_f.flush()
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

        if iter_num >= args.max_iters:
            save_ckpt(model, optimizer, iter_num, best_val_loss, out_dir, "ckpt_final")
            log(f"done: reached max_iters={args.max_iters}, best_val={best_val_loss:.4f}, "
                f"final ckpt saved. total {time.time()-t0:.1f}s this session")
            break

        try:
            logits, loss = model(X, Y)
            X, Y = get_batch("train", *ctx)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        except KeyboardInterrupt:
            save_ckpt(model, optimizer, iter_num, best_val_loss, out_dir, "ckpt_latest")
            log(f"interrupted at iter {iter_num} -- ckpt_latest.pt saved, rerun same command to resume")
            raise

        iter_num += 1
        local_iter += 1

        if iter_num % args.ckpt_every == 0:
            save_ckpt(model, optimizer, iter_num, best_val_loss, out_dir, "ckpt_latest")
            log(f"iter {iter_num:5d}: checkpoint saved ({(time.time()-t0)/local_iter*1000:.0f} ms/iter avg)")

    # final mixing sanity report
    report = model.mixing_report()
    (out_dir / "mixing_report.json").write_text(json.dumps(report, indent=2))
    worst = max(r["row_err"] for r in report)
    log(f"mixing report: worst row_err={worst:.2e} over {len(report)} mixers -> mixing_report.json")


if __name__ == "__main__":
    main()
