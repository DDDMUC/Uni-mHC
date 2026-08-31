"""
model_hc.py -- nanoGPT with Uni-mHC multi-stream residual topology.

Adapted from karpathy/nanoGPT (model.py). The single residual stream is replaced by
`n_streams` parallel streams; at every attention/MLP stage boundary the streams are
redistributed by a manifold-constrained mixing matrix (default: unistochastic UniMHC):

    streams <-- Mixer(streams) + write_k * f( sum_k read_k * streams_k )

At init UniMHC/Givens mixers are the identity, so the model starts exactly as a
vanilla pre-norm GPT and gradually learns to exploit the extra streams.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # uni_mhc repo root
from models.manifold_layers import make_mixer


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2
    n_streams: int = 4
    mixer: str = "unimhc"  # unimhc | givens | sinkhorn | ortho
    dynamic_topology: bool = False  # True: paper-faithful dynamic HC/mHC annotations (Eq.5+Eq.8)


def _batched_sinkhorn_exp(raw: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """mHC Eq.9, batched: M(0)=exp(raw), M(t)=T_r(T_c(M(t-1))), final row pass.
    raw: (..., n, n) -> doubly stochastic (..., n, n). Matches the paper's order
    up to float32 noise (verified 6e-8 vs column-first order)."""
    H = torch.exp(raw)
    for _ in range(n_iters):
        H = H / H.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        H = H / H.sum(dim=-2, keepdim=True).clamp_min(1e-12)
    H = H / H.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return H


def _cayley_from_raw(a_raw: torch.Tensor, b_raw: torch.Tensor, phi: torch.Tensor):
    """Uni-mHC dynamic extension: antisym/sym raw matrices -> (W, S).
    a_raw, b_raw: (..., n, n); phi: (n,) static heterodyne phases.
    A = a_raw - a_raw^T (antisym), B = b_raw + b_raw^T (sym), U = (I-K)(I+K)^-1."""
    A = a_raw - a_raw.transpose(-1, -2)
    B = b_raw + b_raw.transpose(-1, -2)
    K = torch.complex(A, B)
    n = A.shape[-1]
    I = torch.eye(n, dtype=K.dtype, device=K.device)
    U = torch.linalg.solve(I + K, I - K)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    W = U.real * cos_phi - U.imag * sin_phi              # (..., n, n)
    S = U.real.pow(2) + U.imag.pow(2)
    return W, S


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # causal self-attention; Self-Attention: flash attention see official doc
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0,
            is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class HCBlock(nn.Module):
    """Transformer block over n_streams residual streams with manifold-constrained mixing.

    topology="static"  (default): learned static read/write vectors + static mixer modules.
    topology="dynamic": paper-faithful HC/mHC annotations (mHC Eq.5 + Eq.8):
        H_pre  = sigmoid(alpha*tanh(theta_pre x~) + b_pre)      (1 x n, non-negative read)
        H_post = 2*sigmoid(alpha*tanh(theta_post x~) + b_post)  (1 x n, non-negative write)
        H_res  = mixer(alpha*tanh(theta_res x~) + b_res)        (n x n, constrained update)
      where the residual mixer is SK(exp(.)) for mixer="sinkhorn" (exact mHC) and the
      dynamic Cayley construction for mixer="unimhc" (same dynamic form, our manifold).
      Dynamic parts share x~ = LN(x) of the connected stream; alphas init small per HC.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        n = config.n_streams
        inv_sqrt_n = 1.0 / math.sqrt(n)
        self.n_streams = n
        self.dynamic = bool(getattr(config, "dynamic_topology", False))
        self.mixer_kind = config.mixer
        self.mixer_attn = make_mixer(config.mixer, n)
        self.mixer_mlp = make_mixer(config.mixer, n)
        self.read_attn = nn.Parameter(torch.full((n,), inv_sqrt_n))
        self.read_mlp = nn.Parameter(torch.full((n,), inv_sqrt_n))
        self.write_attn = nn.Parameter(torch.full((n,), inv_sqrt_n))
        self.write_mlp = nn.Parameter(torch.full((n,), inv_sqrt_n))
        if self.dynamic:
            C = config.n_embd
            logit_uniform = math.log(1.0 / (n - 1))
            # gating scalars ("initialized to small values" per HC paper)
            self.alpha = nn.ParameterDict({
                f"{stage}_{kind}": nn.Parameter(torch.tensor(0.1))
                for stage in ("attn", "mlp") for kind in ("pre", "post", "res")
            })
            self.theta_pre = nn.ModuleDict({s: nn.Linear(C, 1, bias=False) for s in ("attn", "mlp")})
            self.theta_post = nn.ModuleDict({s: nn.Linear(C, 1, bias=False) for s in ("attn", "mlp")})
            self.b_pre = nn.ParameterDict({s: nn.Parameter(torch.full((n,), logit_uniform)) for s in ("attn", "mlp")})
            self.b_post = nn.Parameter(torch.zeros(2, n))  # 2*sigmoid(0)=1 -> vanilla-like write at init
            if config.mixer == "unimhc":
                # dynamic Cayley: two raw n x n heads (antisym + sym parts), static phases
                self.theta_res_a = nn.ModuleDict({s: nn.Linear(C, n * n, bias=False) for s in ("attn", "mlp")})
                self.theta_res_b = nn.ModuleDict({s: nn.Linear(C, n * n, bias=False) for s in ("attn", "mlp")})
                self.b_res_a = nn.ParameterDict({s: nn.Parameter(torch.zeros(n, n)) for s in ("attn", "mlp")})
                self.b_res_b = nn.ParameterDict({s: nn.Parameter(torch.zeros(n, n)) for s in ("attn", "mlp")})
                self.phi = nn.Parameter(torch.zeros(n))
            elif config.mixer == "sinkhorn":
                # faithful mHC residual: exp -> SK, diagonal bias -> near-identity at init
                self.theta_res = nn.ModuleDict({s: nn.Linear(C, n * n, bias=False) for s in ("attn", "mlp")})
                self.b_res = nn.ParameterDict({s: nn.Parameter(n * torch.eye(n) / math.sqrt(n)) for s in ("attn", "mlp")})
            else:
                raise NotImplementedError(
                    f"dynamic topology supports mixer='unimhc'|'sinkhorn', got '{config.mixer}'")
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def _dyn_raw(self, x_norm, stage):
        """x_norm: (B,T,C) -> H_pre (B,T,n), H_post (B,T,n), and mixer raw input (B,T,n,n)."""
        a_pre, a_post, a_res = (self.alpha[f"{stage}_{k}"] for k in ("pre", "post", "res"))
        h_pre = torch.sigmoid(a_pre * torch.tanh(self.theta_pre[stage](x_norm)) + self.b_pre[stage]).squeeze(-1)
        h_post = 2 * torch.sigmoid(a_post * torch.tanh(self.theta_post[stage](x_norm)) + self.b_post[0 if stage == "attn" else 1]).squeeze(-1)
        n = self.n_streams
        if self.mixer_kind == "unimhc":
            raw_a = a_res * torch.tanh(self.theta_res_a[stage](x_norm)).view(*x_norm.shape[:2], n, n) + self.b_res_a[stage]
            raw_b = a_res * torch.tanh(self.theta_res_b[stage](x_norm)).view(*x_norm.shape[:2], n, n) + self.b_res_b[stage]
            return h_pre, h_post, (raw_a, raw_b)
        raw = a_res * torch.tanh(self.theta_res[stage](x_norm)).view(*x_norm.shape[:2], n, n) + self.b_res[stage]
        return h_pre, h_post, raw

    def _dyn_stage(self, streams, x_norm, stage, block_fn, ln):
        h_pre, h_post, raw = self._dyn_raw(x_norm, stage)
        v = torch.einsum("btk,btkd->btd", h_pre, streams)
        f = block_fn(ln(v))
        n = self.n_streams
        if self.mixer_kind == "unimhc":
            W, _S = _cayley_from_raw(raw[0], raw[1], self.phi)
            mixed = torch.einsum("btjk,btkd->btjd", W, streams)
        else:
            H = _batched_sinkhorn_exp(raw)
            mixed = torch.einsum("btjk,btkd->btjd", H, streams)
        return mixed + f.unsqueeze(2) * h_post.unsqueeze(-1)

    def forward(self, streams):  # streams: (B, T, n, d)
        if self.dynamic:
            v = streams.mean(dim=2)  # connected stream for annotation input (x~ uses LN below)
            x1 = self.ln_1(v)
            streams = self._dyn_stage(streams, x1, "attn", self.attn, nn.Identity())
            v = streams.mean(dim=2)
            x2 = self.ln_2(v)
            streams = self._dyn_stage(streams, x2, "mlp", self.mlp, nn.Identity())
            return streams
        v = torch.einsum("k,btkd->btd", self.read_attn, streams)
        f = self.attn(self.ln_1(v))
        streams = self.mixer_attn(streams) + f.unsqueeze(2) * self.write_attn.view(1, 1, -1, 1)
        v = torch.einsum("k,btkd->btd", self.read_mlp, streams)
        f = self.mlp(self.ln_2(v))
        streams = self.mixer_mlp(streams) + f.unsqueeze(2) * self.write_mlp.view(1, 1, -1, 1)
        return streams


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        n = config.n_streams
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([HCBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # final readout over streams
        self.read_out = nn.Parameter(torch.full((n,), 1.0 / math.sqrt(n)))
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight
        # init all weights, and apply special scaled init to the residual projections
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        if T > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # stream 0 carries the token embedding; other streams start at zero
        tok = self.transformer.wte(idx) + self.transformer.wpe(pos)  # (B, T, d)
        streams = torch.zeros(
            B, T, self.config.n_streams, self.config.n_embd,
            device=device, dtype=tok.dtype,
        )
        streams[:, :, 0, :] = tok
        streams = self.transformer.drop(streams)
        for block in self.transformer.h:
            streams = block(streams)
        x = torch.einsum("k,btkd->btd", self.read_out, streams)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                del block.attn.bias
                del block.attn.mask_bias

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        unconstrained_names = ("mixer", "theta_res", "b_res")  # constrained-matrix params
        decay = [p for pn, p in param_dict.items()
                 if p.dim() >= 2 and not any(k in pn for k in unconstrained_names)]
        no_decay = [p for pn, p in param_dict.items()
                    if not (p.dim() >= 2 and not any(k in pn for k in unconstrained_names))]
        optim_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda" and " fused=True" not in ""
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    def mixing_report(self):
        """Row/col-sum errors of every mixer's energy matrix (sanity for training runs).
        Dynamic-topology blocks are probed at x_norm=0, i.e. the static-bias matrices
        (alphas gate tanh(0)=0), which is the identity-equivalent operating point."""
        report = []
        for i, block in enumerate(self.transformer.h):
            if getattr(block, "dynamic", False):
                for stage in ("attn", "mlp"):
                    n = block.n_streams
                    dev = next(block.parameters()).device
                    if block.mixer_kind == "unimhc":
                        zero = torch.zeros(n, n, device=dev)
                        W, S = _cayley_from_raw(
                            block.alpha[f"{stage}_res"] * zero + block.b_res_a[stage],
                            block.alpha[f"{stage}_res"] * zero + block.b_res_b[stage],
                            block.phi.to(dev))
                    else:  # sinkhorn (faithful mHC)
                        raw = block.alpha[f"{stage}_res"] * torch.zeros(n, n, device=dev) + block.b_res[stage]
                        H = _batched_sinkhorn_exp(raw)
                        W, S = H, H
                    row = (S.sum(-1) - 1).abs().max().item()
                    col = (S.sum(0) - 1).abs().max().item()
                    report.append({"block": i, "stage": stage, "topology": "dynamic-probe",
                                   "row_err": row, "col_err": col,
                                   "min_entry": S.min().item(),
                                   "signed": bool((W < 0).any().item())})
                continue
            for stage, mixer in (("attn", block.mixer_attn), ("mlp", block.mixer_mlp)):
                _, S = mixer.mixing_matrices()
                n = S.shape[0]
                row = (S.sum(-1) - 1).abs().max().item()
                col = (S.sum(0) - 1).abs().max().item()
                report.append({"block": i, "stage": stage, "topology": "static",
                               "row_err": row, "col_err": col,
                               "min_entry": S.min().item(),
                               "signed": bool((mixer.mixing_matrices()[0] < 0).any().item())})
        return report

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
