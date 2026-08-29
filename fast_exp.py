"""
fast_exp.py -- Uni-mHC operator validation + subtractive-disentanglement toy study.

Part A: mathematical precision (doubly-stochastic row/col errors over 100 random draws)
        and speed (1000 forward+backward iterations, ms + peak CUDA memory).
Part B: subtractive disentanglement: target Y = X_0 - X_1. Entrywise-nonnegative mixers
        (Sinkhorn mHC, orthostochastic readout) provably plateau at loss = 1.0
        (= 0.5 * Var(Y)); UniMHC / GivensMHC reach ~0 via phase cancellation (Delta theta = pi).

Outputs: results/bench.json, results/toy.json, figures/disentanglement_loss.png
Usage:   python fast_exp.py [--device cuda] [--skip_bench]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from models.manifold_layers import (
    GivensMHCResidual,
    OrthostochasticMHCResidual,
    SinkhornMHCResidual,
    UniMHCResidual,
    doubly_stochastic_error,
    make_mixer,
)

ROOT = Path(__file__).resolve().parent
METHODS = [
    ("UniMHC (Ours)", "unimhc", "#d62728"),
    ("GivensMHC (Ours-Fast)", "givens", "#ff9896"),
    ("Sinkhorn mHC (Baseline)", "sinkhorn", "#1f77b4"),
    ("Orthostochastic (Baseline)", "ortho", "#7f7f7f"),
]
N_STREAMS = 4


# ===========================================================================
# Part A: precision + latency benchmarks
# ===========================================================================


def bench_precision(device: str, n_draws: int = 100) -> dict:
    """Row/col-sum errors of the energy matrix S over random parameter draws."""
    out = {}
    torch.manual_seed(1337)
    for name, kind, _ in METHODS:
        mixer = make_mixer(kind, N_STREAMS).to(device)
        worst = {"row_err": 0.0, "col_err": 0.0, "min_entry": float("inf")}
        for draw in range(n_draws):
            with torch.no_grad():
                for p in mixer.parameters():
                    p.copy_(torch.randn_like(p) * (1.0 + draw / n_draws))
                _, S = mixer.mixing_matrices()
                err = doubly_stochastic_error(S)
            worst["row_err"] = max(worst["row_err"], err["row_err"])
            worst["col_err"] = max(worst["col_err"], err["col_err"])
            worst["min_entry"] = min(worst["min_entry"], err["min_entry"])
        worst["n_params"] = mixer.num_params()
        out[name] = worst
    return out


def bench_latency(device: str, n_iters: int = 1000) -> dict:
    """1000x forward+backward on GPT-shaped streams (B=32, T=256, n=4, d=384)."""
    out = {}
    B, T, n, d = 32, 256, N_STREAMS, 384
    x = torch.randn(B, T, n, d, device=device)
    for name, kind, _ in METHODS:
        mixer = make_mixer(kind, n).to(device)
        opt = torch.optim.SGD(mixer.parameters(), lr=0.0)
        # warmup
        for _ in range(20):
            loss = mixer(x).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(n_iters):
            loss = mixer(x).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
            wall = (time.perf_counter() - start) * 1e3
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        else:
            wall = (time.perf_counter() - start) * 1e3
            peak_mb = 0.0
        out[name] = {
            "total_ms": round(wall, 1),
            "per_iter_ms": round(wall / n_iters, 3),
            "peak_mem_mb": round(peak_mb, 1),
        }
    return out


# ===========================================================================
# Part B: subtractive disentanglement toy experiment
# ===========================================================================


def disentanglement(device: str, steps: int = 800, batch: int = 512, dim: int = 64) -> dict:
    """Target Y = X_0 - X_1 must be extracted by the *mixing layer alone*.

    Readout: y_hat = gamma * sum_k W[0,k] X_k  (gamma: learnable scalar, same for all).
    For nonnegative W with row-sum 1 the optimum is c = gamma*h = e_0 (all same sign),
    giving floor loss E|X_1|^2 = 1.0 = 0.5 * Var(Y). UniMHC/Givens can place
    opposite-sign coefficients (+1, -1)/sqrt(2) * sqrt(2) and reach 0.
    """
    results = {}
    for name, kind, _ in METHODS:
        torch.manual_seed(1337)
        mixer = make_mixer(kind, N_STREAMS).to(device)
        gamma = torch.nn.Parameter(torch.ones((), device=device))
        opt = torch.optim.Adam(list(mixer.parameters()) + [gamma], lr=5e-2)
        curve = []
        for step in range(steps):
            X = torch.randn(batch, N_STREAMS, dim, device=device)  # fresh data
            W, _ = mixer.mixing_matrices()
            y_hat = gamma * torch.einsum("k,bkd->bd", W[0], X)
            Y = X[:, 0] - X[:, 1]
            loss = torch.nn.functional.mse_loss(y_hat, Y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            curve.append(loss.item())
        W_final, S_final = mixer.mixing_matrices()
        coeffs = (gamma.detach() * W[0]).detach().cpu().tolist()
        # gradient flow sanity
        grads_ok = all(
            (p.grad is None) or torch.isfinite(p.grad).all().item()
            for p in mixer.parameters()
        )
        results[name] = {
            "final_loss": curve[-1],
            "min_loss": min(curve),
            "curve": curve,
            "coeffs": coeffs,  # effective signed mixing coefficients c_k = gamma * W[0,k]
            "gamma": gamma.item(),
            "ds_err": doubly_stochastic_error(S_final.detach()),
            "grads_finite": grads_ok,
        }
    return results


# ===========================================================================
# plotting + main
# ===========================================================================


def plot(results: dict, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    for name, _, color in METHODS:
        ax.plot(results[name]["curve"], label=name, color=color, lw=1.8)
    ax.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.6)
    ax.text(0.02, 1.06, "nonnegative floor = 1.0", fontsize=8, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("MSE loss (log scale)")
    ax.set_title("(a) Subtractive disentanglement:  $Y = X_0 - X_1$")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    width = 0.2
    ks = torch.arange(N_STREAMS)
    for i, (name, _, color) in enumerate(METHODS):
        ax.bar(ks + (i - 1.5) * width, results[name]["coeffs"], width, label=name, color=color)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(ks)
    ax.set_xticklabels([f"$c_{{{k}}}$" for k in range(N_STREAMS)])
    ax.set_ylabel("learned coefficient  $\\gamma\\,W_{0k}$")
    ax.set_title("(b) Final effective mixing row  (target: $+1,-1,0,0$)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Uni-mHC: quantum-interference (destructive) feature subtraction", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip_bench", action="store_true")
    args = ap.parse_args()
    device = args.device
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)

    print(f"device = {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    payload = {"n_streams": N_STREAMS, "device": device}

    if not args.skip_bench:
        print("\n=== Part A1: doubly-stochastic precision (100 random draws) ===")
        precision = bench_precision(device)
        for name, v in precision.items():
            print(
                f"  {name:32s} row_err={v['row_err']:.2e}  col_err={v['col_err']:.2e}  "
                f"min_entry={v['min_entry']:.2e}  params={v['n_params']}"
            )
        print("\n=== Part A2: 1000x forward+backward (B=32,T=256,n=4,d=384) ===")
        latency = bench_latency(device)
        for name, v in latency.items():
            print(
                f"  {name:32s} {v['total_ms']:8.1f} ms total  "
                f"({v['per_iter_ms']:.3f} ms/iter)  peak {v['peak_mem_mb']} MB"
            )
        payload["precision"] = precision
        payload["latency"] = latency
        (ROOT / "results" / "bench.json").write_text(json.dumps(payload, indent=2))

    print("\n=== Part B: subtractive disentanglement ===")
    toy = disentanglement(device)
    for name, v in toy.items():
        print(
            f"  {name:32s} final={v['final_loss']:.3e}  min={v['min_loss']:.3e}  "
            f"coeffs=[{', '.join(f'{c:+.3f}' for c in v['coeffs'])}]  "
            f"grads_ok={v['grads_finite']}"
        )
    payload["toy"] = {
        name: {k: v for k, v in r.items() if k != "curve"} for name, r in toy.items()
    }
    payload["toy_curves"] = {name: r["curve"] for name, r in toy.items()}
    (ROOT / "results" / "toy.json").write_text(json.dumps(payload, indent=2))

    out_png = ROOT / "figures" / "disentanglement_loss.png"
    plot(toy, out_png)
    print(f"\nfigure saved: {out_png}")


if __name__ == "__main__":
    main()
