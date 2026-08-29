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
    """Transformer block over n_streams residual streams with manifold-constrained mixing."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        n = config.n_streams
        inv_sqrt_n = 1.0 / math.sqrt(n)
        self.mixer_attn = make_mixer(config.mixer, n)
        self.mixer_mlp = make_mixer(config.mixer, n)
        self.read_attn = nn.Parameter(torch.full((n,), inv_sqrt_n))
        self.read_mlp = nn.Parameter(torch.full((n,), inv_sqrt_n))
        self.write_attn = nn.Parameter(torch.full((n,), inv_sqrt_n))
        self.write_mlp = nn.Parameter(torch.full((n,), inv_sqrt_n))
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, streams):  # streams: (B, T, n, d)
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
        decay = [p for pn, p in param_dict.items()
                 if p.dim() >= 2 and "mixer" not in pn]
        no_decay = [p for pn, p in param_dict.items()
                    if not (p.dim() >= 2 and "mixer" not in pn)]
        optim_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda" and " fused=True" not in ""
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    def mixing_report(self):
        """Row/col-sum errors of every mixer's energy matrix (sanity for training runs)."""
        report = []
        for i, block in enumerate(self.transformer.h):
            for stage, mixer in (("attn", block.mixer_attn), ("mlp", block.mixer_mlp)):
                _, S = mixer.mixing_matrices()
                n = S.shape[0]
                row = (S.sum(-1) - 1).abs().max().item()
                col = (S.sum(0) - 1).abs().max().item()
                report.append({"block": i, "stage": stage,
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
