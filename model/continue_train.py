"""
Continued training (annealing) of best_model1.pt on the curated corpus.

Mix: each sequence comes from the focus pool with probability --focus-frac, else general.
LR: linear warmup to --peak-lr, cosine decay to --min-lr.
Evaluates on focus val, general val, and the old corpus val split (forgetting check).
Never touches best_model1.pt; writes out/anneal_last.pt, out/anneal_best.pt, out/anneal_log.csv.

Usage:
    python continue_train.py            # resumes automatically from out/anneal_last.pt
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from train import Configuration, Model, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
CURATED = ROOT / "data" / "processed" / "curated"


def get_lr(it, args):
    if it < args.warmup_iters:
        return args.peak_lr * (it + 1) / args.warmup_iters
    ratio = min(1.0, (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters))
    return args.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (args.peak_lr - args.min_lr)


def sample_rows(data, n, block_size, rng):
    ix = rng.integers(0, len(data) - block_size - 1, size=n)
    return np.stack([data[i: i + block_size + 1] for i in ix]).astype(np.int64)


class ChunkSampler:
    """Walks a pool's 1024-token chunks in a shuffled order, so no chunk repeats within an epoch.

    The order is regenerated from (seed, epoch), so only (epoch, pos) needs checkpointing.
    """

    def __init__(self, data, block_size, seed, epoch=0, pos=0):
        self.data, self.block, self.seed = data, block_size, seed
        self.n_chunks = (len(data) - 1) // block_size
        self.epoch, self.pos = epoch, pos
        self.order = np.random.default_rng((seed, epoch)).permutation(self.n_chunks)

    def take(self, n):
        rows = []
        for _ in range(n):
            if self.pos == self.n_chunks:
                self.epoch, self.pos = self.epoch + 1, 0
                self.order = np.random.default_rng((self.seed, self.epoch)).permutation(self.n_chunks)
            start = self.order[self.pos] * self.block
            rows.append(self.data[start: start + self.block + 1])
            self.pos += 1
        return np.stack(rows).astype(np.int64)

    def state(self):
        return {"epoch": self.epoch, "pos": self.pos}


def to_device(rows, device):
    t = torch.from_numpy(rows).pin_memory().to(device, non_blocking=True)
    return t[:, :-1], t[:, 1:]


def get_mixed_rows(samplers, args, rng):
    n_focus = int((rng.random(args.batch_size) < args.focus_frac).sum())
    parts = []
    if n_focus:
        parts.append(samplers["focus"].take(n_focus))
    if args.batch_size - n_focus:
        parts.append(samplers["general"].take(args.batch_size - n_focus))
    return np.concatenate(parts)


def get_mixed_batch(samplers, args, rng, device):
    return to_device(get_mixed_rows(samplers, args, rng), device)


@torch.no_grad()
def evaluate(model, data, args, block_size, device):
    model.eval()
    rng = np.random.default_rng(1234)  # same batches every eval, so curves are comparable
    total = 0.0
    for _ in range(args.eval_iters):
        x, y = to_device(sample_rows(data, args.batch_size, block_size, rng), device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, target=y)
        total += loss.item()
    model.train()
    return total / args.eval_iters


def main():
    parser = argparse.ArgumentParser(description="Continue training on the curated corpus.")
    parser.add_argument("--init-ckpt", default=str(OUT / "best_model1.pt"))
    parser.add_argument("--max-iters", type=int, default=20000)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--peak-lr", type=float, default=1e-5, help="Old run had decayed to ~4e-5 at iter 88k")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--focus-frac", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch; 8 overflows 8 GB VRAM")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--ckpt-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true", help="torch.compile the model (needs triton)")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    config = Configuration()
    block = config.block_size

    focus = np.memmap(CURATED / "focus_train.bin", dtype=np.uint16, mode="r")
    general = np.memmap(CURATED / "general_train.bin", dtype=np.uint16, mode="r")
    val_sets = {
        "focus_val": np.memmap(CURATED / "focus_val.bin", dtype=np.uint16, mode="r"),
        "general_val": np.memmap(CURATED / "general_val.bin", dtype=np.uint16, mode="r"),
        # last 20% of the old corpus bin (the split train.py used for validation), saved as its own file
        "old_val": np.memmap(CURATED / "old_val.bin", dtype=np.uint16, mode="r"),
    }
    print(f"focus_train {len(focus):,} | general_train {len(general):,} tokens; "
          + " | ".join(f"{k} {len(v):,}" for k, v in val_sets.items()))

    model = Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, fused=True)
    last_path, best_path, log_path = OUT / "anneal_last.pt", OUT / "anneal_best.pt", OUT / "anneal_log.csv"

    start_iter, best_focus, sampler_state = 0, float("inf"), {}
    if last_path.exists():
        ckpt = load_checkpoint(last_path, device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter, best_focus = ckpt["iter_num"] + 1, ckpt["best_val_loss"]
        sampler_state = ckpt.get("samplers", {})
        print(f"resumed from {last_path} at iter {start_iter}")
    else:
        model.load_state_dict(load_checkpoint(args.init_ckpt, device)["model"])
        print(f"initialized from {args.init_ckpt} (fresh optimizer)")
        log_path.write_text("iter,lr," + ",".join(val_sets) + "\n")

    samplers = {name: ChunkSampler(data, block, seed=args.seed + k, **sampler_state.get(name, {}))
                for k, (name, data) in enumerate([("focus", focus), ("general", general)])}
    raw_model = model  # saved without the "_orig_mod." prefix torch.compile adds
    if args.compile:
        model = torch.compile(model)

    def save(path, it):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "iter_num": it, "best_val_loss": best_focus, "args": vars(args),
                    "samplers": {k: s.state() for k, s in samplers.items()}}, tmp)
        os.replace(tmp, path)  # atomic: a crash mid-save never corrupts the previous checkpoint

    rng = np.random.default_rng(args.seed + start_iter)
    tokens_per_iter = args.batch_size * args.grad_accum * block
    model.train()
    t0, last_print = time.time(), start_iter - 1
    for it in range(start_iter, args.max_iters + 1):
        lr = get_lr(it, args)
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = {k: evaluate(model, v, args, block, device) for k, v in val_sets.items()}
            print(f"=== iter {it} | " + " | ".join(f"{k} {v:.4f}" for k, v in losses.items()), flush=True)
            with log_path.open("a") as f:
                f.write(f"{it},{lr:.3e}," + ",".join(f"{v:.4f}" for v in losses.values()) + "\n")
            if losses["focus_val"] < best_focus and it > start_iter:
                best_focus = losses["focus_val"]
                save(best_path, it)
            if it == args.max_iters:
                break

        for group in optimizer.param_groups:
            group["lr"] = lr
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_mixed_batch(samplers, args, rng, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, target=y)
            loss = loss / args.grad_accum
            accum_loss += loss.item()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if it % 50 == 0:
            dt = time.time() - t0
            rate = tokens_per_iter * (it - last_print) / dt
            t0, last_print = time.time(), it
            print(f"iter {it} | loss {accum_loss:.4f} | lr {lr:.2e} | {rate:,.0f} tok/s", flush=True)
        if it % args.ckpt_interval == 0 and it > start_iter:
            save(last_path, it)

    save(last_path, args.max_iters)
    print(f"done; best focus_val {best_focus:.4f}; "
          + " | ".join(f"{k} epoch {s.epoch} + {s.pos / s.n_chunks:.0%}" for k, s in samplers.items()))


if __name__ == "__main__":
    main()
