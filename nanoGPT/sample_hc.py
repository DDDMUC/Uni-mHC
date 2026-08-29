"""sample_hc.py -- generate text from a trained HC model checkpoint."""
import argparse
import pickle
from pathlib import Path

import torch

from model_hc import GPT, GPTConfig

p = argparse.ArgumentParser()
p.add_argument("--ckpt", type=Path, required=True)
p.add_argument("--out", type=Path, default=None)
p.add_argument("--max_new_tokens", type=int, default=500)
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--top_k", type=int, default=40)
p.add_argument("--prompt", type=str, default="\n")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
state = torch.load(args.ckpt, map_location=device, weights_only=False)
conf = {k: v for k, v in state["model_args"].items()}
model = GPT(GPTConfig(**conf)).to(device)
model.load_state_dict(state["model"])
model.eval()

meta_path = Path(__file__).resolve().parent / "data/shakespeare_char/meta.pkl"
with open(meta_path, "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

x = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
text = decode(y[0].tolist())
print(f"--- sample from {args.ckpt.name} (iter {state.get('iter_num')}, "
      f"best_val {state.get('best_val_loss'):.4f}) ---")
print(text)
if args.out:
    args.out.write_text(text, encoding="utf-8")
    print(f"saved: {args.out}")
