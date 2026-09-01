"""
model_llama_hc.py -- Llama-3-style backbone with Uni-mHC multi-stream manifold mixing.

Backbone upgrades vs nanoGPT (GPT-2 style), all standard Llama 3 practice:
  * RMSNorm (no mean subtraction, no bias)
  * RoPE rotary position embeddings (HF rotate_half convention) -- no learned wpe
  * SwiGLU MLP (silu gate) with multiple-of rounding
  * Grouped-Query Attention (n_kv_heads <= n_heads)
  * no biases anywhere; untied lm_head

Uni-mHC machinery is identical to model_hc.py: n parallel residual streams,
manifold mixers at every stage boundary (static topology, 4 mixers), or the
paper-faithful dynamic topology (mHC Eq.5+Eq.8) for mixer in {unimhc, sinkhorn}.
Stream 0 carries the token embedding at init; streams 1..n-1 start at zero.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.manifold_layers import make_mixer

from model_hc import _batched_sinkhorn_exp, _cayley_from_raw


@dataclass
class LlamaHCConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_kv_heads: int = 0          # 0 -> n_head (plain MHA); must divide n_head
    n_embd: int = 384
    multiple_of: int = 64        # SwiGLU hidden dim rounding
    ffn_dim_multiplier: float = 1.0
    rope_theta: float = 500000.0  # Llama 3 value (Llama 2 used 10000.0)
    dropout: float = 0.0
    n_streams: int = 4
    mixer: str = "unimhc"        # unimhc | givens | sinkhorn | ortho
    dynamic_topology: bool = False

    def __post_init__(self):
        if self.n_kv_heads in (0, None):
            self.n_kv_heads = self.n_head
        assert self.n_head % self.n_kv_heads == 0, "n_kv_heads must divide n_head"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # float32 cast per reference implementations (matters under bf16/fp16 autocast)
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).type_as(x)


def precompute_rope(head_dim: int, max_len: int, theta: float, device):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)).to(device)
    t = torch.arange(max_len, device=device)
    freqs = torch.outer(t, freqs)                      # (T, hs/2)
    emb = torch.cat((freqs, freqs), dim=-1)            # (T, hs), HF convention
    return emb.cos().to(torch.float32), emb.sin().to(torch.float32)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, n_h, T, hs); cos/sin: (T, hs)
    T = x.shape[2]
    return x * cos[None, None, :T] + rotate_half(x) * sin[None, None, :T]


def repeat_kv(x: torch.Tensor, rep: int) -> torch.Tensor:
    # (B, n_kv, T, hs) -> (B, n_kv*rep, T, hs)
    if rep == 1:
        return x
    B, nk, T, hs = x.shape
    return x[:, :, None].expand(B, nk, rep, T, hs).reshape(B, nk * rep, T, hs)


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaHCConfig):
        super().__init__()
        d, nh, nkv = config.n_embd, config.n_head, config.n_kv_heads
        hs = d // nh
        self.nh, self.nkv, self.hs = nh, nkv, hs
        self.rep = nh // nkv
        self.wq = nn.Linear(d, nh * hs, bias=False)
        self.wk = nn.Linear(d, nkv * hs, bias=False)
        self.wv = nn.Linear(d, nkv * hs, bias=False)
        self.wo = nn.Linear(nh * hs, d, bias=False)
        self.dropout = config.dropout

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hs).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, self.hs).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, self.hs).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k, v = repeat_kv(k, self.rep), repeat_kv(v, self.rep)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.wo(y)


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaHCConfig):
        super().__init__()
        d = config.n_embd
        hidden = int(2 * d * 4 / 3)
        hidden = config.multiple_of * ((hidden + config.multiple_of - 1) // config.multiple_of)
        hidden = int(config.ffn_dim_multiplier * hidden)
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class LlamaHCBlock(nn.Module):
    """Two Uni-mHC stages (attn, mlp) around Llama-style sublayers."""

    def __init__(self, config: LlamaHCConfig):
        super().__init__()
        n = config.n_streams
        inv_sqrt_n = 1.0 / math.sqrt(n)
        self.n_streams = n
        self.dynamic = bool(config.dynamic_topology)
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
            self.alpha = nn.ParameterDict({
                f"{s}_{k}": nn.Parameter(torch.tensor(0.1))
                for s in ("attn", "mlp") for k in ("pre", "post", "res")
            })
            self.theta_pre = nn.ModuleDict({s: nn.Linear(C, 1, bias=False) for s in ("attn", "mlp")})
            self.theta_post = nn.ModuleDict({s: nn.Linear(C, 1, bias=False) for s in ("attn", "mlp")})
            self.b_pre = nn.ParameterDict({s: nn.Parameter(torch.full((n,), logit_uniform)) for s in ("attn", "mlp")})
            self.b_post = nn.Parameter(torch.zeros(2, n))
            if config.mixer == "unimhc":
                self.theta_res_a = nn.ModuleDict({s: nn.Linear(C, n * n, bias=False) for s in ("attn", "mlp")})
                self.theta_res_b = nn.ModuleDict({s: nn.Linear(C, n * n, bias=False) for s in ("attn", "mlp")})
                self.b_res_a = nn.ParameterDict({s: nn.Parameter(torch.zeros(n, n)) for s in ("attn", "mlp")})
                self.b_res_b = nn.ParameterDict({s: nn.Parameter(torch.zeros(n, n)) for s in ("attn", "mlp")})
                self.phi = nn.Parameter(torch.zeros(n))
            elif config.mixer == "sinkhorn":
                self.theta_res = nn.ModuleDict({s: nn.Linear(C, n * n, bias=False) for s in ("attn", "mlp")})
                self.b_res = nn.ParameterDict({s: nn.Parameter(n * torch.eye(n) / math.sqrt(n)) for s in ("attn", "mlp")})
            else:
                raise NotImplementedError("dynamic topology supports mixer='unimhc'|'sinkhorn'")
        self.norm_attn = RMSNorm(config.n_embd)
        self.attn = LlamaAttention(config)
        self.norm_mlp = RMSNorm(config.n_embd)
        self.mlp = LlamaMLP(config)

    def _dyn_raw(self, x_norm, stage):
        a_pre, a_post, a_res = (self.alpha[f"{stage}_{k}"] for k in ("pre", "post", "res"))
        h_pre = torch.sigmoid(a_pre * torch.tanh(self.theta_pre[stage](x_norm)) + self.b_pre[stage]).squeeze(-1)
        h_post = 2 * torch.sigmoid(a_post * torch.tanh(self.theta_post[stage](x_norm)) + self.b_post[0 if stage == "attn" else 1]).squeeze(-1)
        n = self.n_streams
        if self.mixer_kind == "unimhc":
            ra = a_res * torch.tanh(self.theta_res_a[stage](x_norm)).view(*x_norm.shape[:2], n, n) + self.b_res_a[stage]
            rb = a_res * torch.tanh(self.theta_res_b[stage](x_norm)).view(*x_norm.shape[:2], n, n) + self.b_res_b[stage]
            return h_pre, h_post, (ra, rb)
        raw = a_res * torch.tanh(self.theta_res[stage](x_norm)).view(*x_norm.shape[:2], n, n) + self.b_res[stage]
        return h_pre, h_post, raw

    def _dyn_stage(self, streams, x_norm, stage, sublayer):
        h_pre, h_post, raw = self._dyn_raw(x_norm, stage)
        v = torch.einsum("btk,btkd->btd", h_pre, streams)
        f = sublayer(v)
        if self.mixer_kind == "unimhc":
            W, _ = _cayley_from_raw(raw[0], raw[1], self.phi)
        else:
            W = _batched_sinkhorn_exp(raw)
        mixed = torch.einsum("btjk,btkd->btjd", W, streams)
        return mixed + f.unsqueeze(2) * h_post.unsqueeze(-1)

    def forward(self, streams, cos, sin):  # streams: (B, T, n, d)
        if self.dynamic:
            v = streams.mean(dim=2)
            x1 = self.norm_attn(v)
            streams = self._dyn_stage(streams, x1, "attn", lambda t: self.attn(t, cos, sin))
            v = streams.mean(dim=2)
            x2 = self.norm_mlp(v)
            streams = self._dyn_stage(streams, x2, "mlp", self.mlp)
            return streams
        v = torch.einsum("k,btkd->btd", self.read_attn, streams)
        f = self.attn(self.norm_attn(v), cos, sin)
        streams = self.mixer_attn(streams) + f.unsqueeze(2) * self.write_attn.view(1, 1, -1, 1)
        v = torch.einsum("k,btkd->btd", self.read_mlp, streams)
        f = self.mlp(self.norm_mlp(v))
        streams = self.mixer_mlp(streams) + f.unsqueeze(2) * self.write_mlp.view(1, 1, -1, 1)
        return streams


class LlamaHC(nn.Module):

    def __init__(self, config: LlamaHCConfig):
        super().__init__()
        assert config.vocab_size is not None and config.block_size is not None
        self.config = config
        n = config.n_streams
        hs = config.n_embd // config.n_head
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([LlamaHCBlock(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))
        cos, sin = precompute_rope(hs, config.block_size, config.rope_theta, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.read_out = nn.Parameter(torch.full((n,), 1.0 / math.sqrt(n)))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        if T > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
        cos = self.rope_cos.to(device)
        sin = self.rope_sin.to(device)

        tok = self.transformer.wte(idx)
        streams = torch.zeros(B, T, self.config.n_streams, self.config.n_embd,
                              device=device, dtype=tok.dtype)
        streams[:, :, 0, :] = tok
        streams = self.transformer.drop(streams)
        for block in self.transformer.h:
            streams = block(streams, cos, sin)
        x = torch.einsum("k,btkd->btd", self.read_out, streams)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            logits = logits[:, [-1], :]
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        unconstrained = ("mixer", "theta_res", "b_res")
        decay = [p for pn, p in param_dict.items()
                 if p.dim() >= 2 and not any(k in pn for k in unconstrained)]
        no_decay = [p for pn, p in param_dict.items()
                    if not (p.dim() >= 2 and not any(k in pn for k in unconstrained))]
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=learning_rate, betas=betas,
            **({"fused": True} if device_type == "cuda" else {}))
        return optimizer

    def mixing_report(self):
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
                    else:
                        raw = block.alpha[f"{stage}_res"] * torch.zeros(n, n, device=dev) + block.b_res[stage]
                        H = _batched_sinkhorn_exp(raw)
                        W, S = H, H
                    report.append({"block": i, "stage": stage, "topology": "dynamic-probe",
                                   "row_err": (S.sum(-1) - 1).abs().max().item(),
                                   "col_err": (S.sum(0) - 1).abs().max().item(),
                                   "min_entry": S.min().item(),
                                   "signed": bool((W < 0).any().item())})
                continue
            for stage, mixer in (("attn", block.mixer_attn), ("mlp", block.mixer_mlp)):
                _, S = mixer.mixing_matrices()
                report.append({"block": i, "stage": stage, "topology": "static",
                               "row_err": (S.sum(-1) - 1).abs().max().item(),
                               "col_err": (S.sum(0) - 1).abs().max().item(),
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
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx
