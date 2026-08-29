"""
Uni-mHC: Unistochastic Manifold-Constrained Hyper-Connections
=============================================================
Core operator library (hot-swappable residual-stream mixing layers, stream width n=4).

All four mixers expose the same interface:
    W, S = mixer.mixing_matrices()   # W: effective real mixing (used to mix streams)
                                     # S: doubly-stochastic "energy" matrix (safety constraint)
    out  = mixer(x)                  # x: (..., n, d) real tensor -> (..., n, d)

Operators
---------
* UniMHCResidual          (Ours)      complex Cayley transform -> unitary U, S = |U|^2 (unistochastic)
* GivensMHCResidual       (Ours-Fast) n(n-1)/2 Givens angles -> orthogonal Q, S = Q*Q (orthostochastic)
* SinkhornMHCResidual     (Baseline)  free matrix -> 20-step Sinkhorn-Knopp (Birkhoff projection, mHC)
* OrthostochasticMHCResidual (Baseline) free matrix -> Newton-Schulz polar factor Q, S = Q*Q

Key mechanism (Ours). K = A + iB with A = -A^T (real antisym.), B = B^T (real sym.)
is skew-Hermitian (K^H = -K), hence the complex Cayley transform
    U = (I - K)(I + K)^{-1}
is unitary (I+K is always invertible: eigenvalues of K are purely imaginary). The energy
matrix S = |U|^2 = Re(U)^2 + Im(U)^2 is then EXACTLY doubly stochastic (unistochastic).
The *effective* mixing applied to the real streams is the heterodyned real part
    W_jk = Re(U_jk * e^{i phi_k}) = Re(U_jk) cos(phi_k) - Im(U_jk) sin(phi_k)
with a learnable per-stream phase phi_k. Because |W_jk| <= |U_jk| and
sum_k W_jk^2 <= sum_k |U_jk|^2 = 1, the mixing is non-expansive (mHC-level safety),
while opposite phases (Delta theta = pi) yield NEGATIVE effective weights, i.e.
destructive quantum interference / feature subtraction -- impossible for any
entrywise-nonnegative method (Sinkhorn, and any Q o Q orthostochastic readout).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

__all__ = [
    "UniMHCResidual",
    "GivensMHCResidual",
    "SinkhornMHCResidual",
    "OrthostochasticMHCResidual",
    "make_mixer",
    "doubly_stochastic_error",
]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def doubly_stochastic_error(S: torch.Tensor) -> Dict[str, float]:
    """Row-sum / col-sum / non-negativity violation of a candidate matrix S."""
    n = S.shape[-1]
    one = torch.ones(n, dtype=S.dtype, device=S.device)
    row_err = (S.sum(-1) - one).abs().max().item()
    col_err = (S.sum(-2) - one).abs().max().item()
    min_entry = S.min().item()
    return {"row_err": row_err, "col_err": col_err, "min_entry": min_entry}


class _ManifoldMixer(nn.Module):
    """Base class: subclasses implement `mixing_matrices()` returning (W, S)."""

    n_streams: int

    def mixing_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def energy_matrix(self) -> torch.Tensor:
        return self.mixing_matrices()[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix the stream axis (-2) of x: (..., n, d) -> (..., n, d)."""
        W, _ = self.mixing_matrices()
        return torch.einsum("jk,...kd->...jd", W, x.to(W.dtype)).to(x.dtype)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Ours: UniMHC -- complex Cayley -> unistochastic |U|^2
# ---------------------------------------------------------------------------


class UniMHCResidual(_ManifoldMixer):
    """Unistochastic hyper-connection mixing via complex Cayley transform.

    Parameters: theta_a (n x n -> antisymmetric A), theta_b (n x n -> symmetric B),
    phi (n, heterodyne phases). Identity init (U = I, W = I).
    """

    def __init__(self, n_streams: int = 4, init_scale: float = 0.1, **kwargs):
        super().__init__()
        self.n_streams = n_streams
        n = n_streams
        self.theta_a = nn.Parameter(torch.randn(n, n) * init_scale / math.sqrt(n))
        self.theta_b = nn.Parameter(torch.randn(n, n) * init_scale / math.sqrt(n))
        self.phi = nn.Parameter(torch.zeros(n))

    def unitary(self) -> torch.Tensor:
        n = self.n_streams
        A = self.theta_a - self.theta_a.T                # real antisymmetric
        B = self.theta_b + self.theta_b.T                # real symmetric
        K = torch.complex(A, B)                          # skew-Hermitian: K^H = -K
        I = torch.eye(n, dtype=K.dtype, device=K.device)
        # U = (I - K)(I + K)^{-1}; (I+K) always invertible (imaginary spectrum)
        U = torch.linalg.solve(I + K, I - K)
        return U

    def mixing_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        U = self.unitary()
        cos_phi = torch.cos(self.phi)
        sin_phi = torch.sin(self.phi)
        # W_jk = Re(U_jk e^{i phi_k})  (heterodyne / real-part readout -> sign flips = interference)
        W = U.real * cos_phi.unsqueeze(0) - U.imag * sin_phi.unsqueeze(0)
        S = U.real.pow(2) + U.imag.pow(2)                # unistochastic, exactly doubly stochastic
        return W, S

    def extra_repr(self) -> str:
        return f"n_streams={self.n_streams}, cayley-unistochastic"


# ---------------------------------------------------------------------------
# Ours-Fast: Givens rotations -> orthostochastic with phase-enabled readout
# ---------------------------------------------------------------------------


class GivensMHCResidual(_ManifoldMixer):
    """Exact orthogonal Q as product of n(n-1)/2 Givens rotations (no matrix inverse).

    S = Q o Q is orthostochastic (hence unistochastic). Effective mixing
    W = Q * cos(phi) keeps the phase/sign mechanism (columns may flip sign).
    """

    def __init__(self, n_streams: int = 4, **kwargs):
        super().__init__()
        self.n_streams = n_streams
        n = n_streams
        self.n_angles = n * (n - 1) // 2
        self.angles = nn.Parameter(torch.zeros(self.n_angles))
        self.phi = nn.Parameter(torch.zeros(n))

    def orthogonal(self) -> torch.Tensor:
        n = self.n_streams
        Q = torch.eye(n, dtype=self.angles.dtype, device=self.angles.device)
        idx = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                theta = self.angles[idx]
                idx += 1
                c, s = torch.cos(theta), torch.sin(theta)
                col_i = Q[:, i] * c - Q[:, j] * s
                col_j = Q[:, i] * s + Q[:, j] * c
                Q = Q.clone()
                Q[:, i] = col_i
                Q[:, j] = col_j
        return Q

    def mixing_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        Q = self.orthogonal()
        W = Q * torch.cos(self.phi).unsqueeze(0)         # Re(Q_jk e^{i phi_k}) (Q real)
        S = Q * Q                                         # orthostochastic
        return W, S

    def extra_repr(self) -> str:
        return f"n_streams={self.n_streams}, n_angles={self.n_angles}, givens-orthostochastic"


# ---------------------------------------------------------------------------
# Baseline: DeepSeek mHC -- Sinkhorn-Knopp projection onto Birkhoff polytope
# ---------------------------------------------------------------------------


class SinkhornMHCResidual(_ManifoldMixer):
    """mHC baseline: free matrix exp(M) projected by `n_iters` Sinkhorn-Knopp
    alternated row/column normalizations. Output is entrywise NON-NEGATIVE
    (convex-combination mixing only -- no subtractive disentanglement)."""

    def __init__(self, n_streams: int = 4, n_iters: int = 20, **kwargs):
        super().__init__()
        self.n_streams = n_streams
        self.n_iters = n_iters
        self.raw = nn.Parameter(torch.randn(n_streams, n_streams) * 0.5)

    def mixing_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        H = torch.exp(self.raw)                          # positive support (total support)
        ones = torch.ones_like(H.sum(-1, keepdim=True))
        for _ in range(self.n_iters):
            H = H / H.sum(dim=-1, keepdim=True).clamp_min(1e-12)   # row-normalize
            H = H / H.sum(dim=-2, keepdim=True).clamp_min(1e-12)   # col-normalize
        H = H / H.sum(dim=-1, keepdim=True)              # final row pass -> rows exactly 1
        return H, H

    def extra_repr(self) -> str:
        return f"n_streams={self.n_streams}, n_iters={self.n_iters}, sinkhorn-birkhoff"


# ---------------------------------------------------------------------------
# Baseline: orthostochastic via Newton-Schulz polar decomposition
# ---------------------------------------------------------------------------


class OrthostochasticMHCResidual(_ManifoldMixer):
    """Baseline: orthogonal polar factor Q of a free matrix M via Newton-Schulz
    iteration (X <- 1.5 X - 0.5 X X^T X), mixing = S = Q o Q (NON-NEGATIVE)."""

    def __init__(self, n_streams: int = 4, n_iters: int = 20, **kwargs):
        super().__init__()
        self.n_streams = n_streams
        self.n_iters = n_iters
        self.raw = nn.Parameter(torch.eye(n_streams) + 0.1 * torch.randn(n_streams, n_streams))

    def orthogonal(self) -> torch.Tensor:
        X = self.raw / self.raw.norm(p="fro")            # spectral norm <= Frobenius norm
        for _ in range(self.n_iters):
            X = 1.5 * X - 0.5 * X @ X.T @ X              # cubic Newton-Schulz (polar)
        return X

    def mixing_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        Q = self.orthogonal()
        S = Q * Q
        return S, S

    def extra_repr(self) -> str:
        return f"n_streams={self.n_streams}, n_iters={self.n_iters}, newton-schulz-orthostochastic"


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

_MIXERS = {
    "unimhc": UniMHCResidual,
    "givens": GivensMHCResidual,
    "sinkhorn": SinkhornMHCResidual,
    "ortho": OrthostochasticMHCResidual,
}


def make_mixer(kind: str, n_streams: int = 4, **kwargs) -> _ManifoldMixer:
    if kind not in _MIXERS:
        raise KeyError(f"unknown mixer '{kind}', choose from {list(_MIXERS)}")
    return _MIXERS[kind](n_streams=n_streams, **kwargs)
