"""replot_figure.py -- regenerate figures/disentanglement_loss.png from results/toy.json
(no experiment re-run; same data as the paper)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
payload = json.loads((ROOT / "results" / "toy.json").read_text())
toy, curves = payload["toy"], payload["toy_curves"]
METHODS = [
    ("UniMHC (Ours)", "#d62728"),
    ("GivensMHC (Ours-Fast)", "#ff9896"),
    ("Sinkhorn mHC (Baseline)", "#1f77b4"),
    ("Orthostochastic (Baseline)", "#7f7f7f"),
]
N = 4

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
for name, color in METHODS:
    ax.plot(curves[name], label=name, color=color, lw=1.8)
ax.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.6)
ax.text(790, 1.35, "nonnegative floor = 1.0", fontsize=8, alpha=0.8, ha="right")
ax.set_yscale("log")
ax.set_xlabel("training step")
ax.set_ylabel("MSE loss (log scale)")
ax.set_title(r"(a) Subtractive disentanglement:  $Y = X_0 - X_1$")
ax.legend(fontsize=8, frameon=False, loc="lower left")
ax.grid(alpha=0.25, which="both")

ax = axes[1]
width = 0.2
for i, (name, color) in enumerate(METHODS):
    ax.bar([k + (i - 1.5) * width for k in range(N)], toy[name]["coeffs"],
           width, label=name, color=color)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(N))
ax.set_xticklabels([rf"$c_{{{k}}}$" for k in range(N)])
ax.set_ylabel(r"learned coefficient  $\gamma\,W_{0k}$")
ax.set_title(r"(b) Final effective mixing row  (target: $+1,-1,0,0$)")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, axis="y")

fig.suptitle("Uni-mHC: quantum-interference (destructive) feature subtraction", y=1.02)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "disentanglement_loss.png", dpi=200, bbox_inches="tight")
print("replotted OK")
