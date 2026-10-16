"""
Pretrain a ~350M model from scratch on the page-cleaned corpus (data/processed/clean).

Same architecture as train.py (RoPE, SwiGLU, RMSNorm), scaled to 24 layers x 1024.
Mix: each sequence comes from the focus pool with probability --focus-frac, else general.
Writes out/pre350_last.pt (every --ckpt-interval), out/pre350_best.pt (best focus_val),
out/pre350_log.csv. Rerunning the same command resumes from pre350_last.pt.

Usage (A100/H100 80GB):
    python pretrain.py --compile
Multi-GPU (same batches and results as one GPU; GPU r computes micro-batches r, r+N, ...):
    torchrun --nproc_per_node 8 pretrain.py --compile
"""

from __future__ import annotations

import argparse
import builtins
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from continue_train import ChunkSampler, evaluate, get_lr, get_mixed_rows, to_device
from train import Configuration, Model, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
CLEAN = ROOT / "data" / "processed" / "clean"
CONFIG_350M = dict(n_embd=1024, n_layer=24, n_head=16, n_swi=2816)


def init_weights(model, config):
    """GPT-2 style init: N(0, 0.02), residual output projections scaled down by depth."""
    for name, p in model.named_parameters():
        if p.dim() == 2:  # the tied embedding/head is visited once, as wte
            std = 0.02 / math.sqrt(2 * config.n_layer) if name.endswith(("lproj.weight", "swiglu.proj.weight")) else 0.02
            torch.nn.init.normal_(p, mean=0.0, std=std)


def main():
    parser = argparse.ArgumentParser(description="Pretrain the 350M model from scratch.")
    parser.add_argument("--max-iters", type=int, default=16000, help="16000 x 0.5M tokens = 8.4B tokens")
    parser.add_argument("--warmup-iters", type=int, default=700)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--focus-frac", type=float, default=0.45)
    parser.add_argument("--batch-size", type=int, default=16, help="Micro-batch (sequences)")
    parser.add_argument("--grad-accum", type=int, default=32, help="16 x 32 x 1024 = 0.5M tokens/iter")
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--ckpt-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--in-memory", action="store_true", help="Load train bins into RAM (network disks)")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ddp = "RANK" in os.environ  # launched by torchrun
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        device = f"cuda:{int(os.environ['LOCAL_RANK']) % torch.cuda.device_count()}"
        torch.cuda.set_device(device)
    else:
        rank, world, device = 0, 1, "cuda"
    master = rank == 0
    assert args.grad_accum % world == 0, "grad_accum must be divisible by the number of GPUs"
    if not master:
        builtins.print = lambda *a, **k: None  # one rank prints
    config = Configuration(**CONFIG_350M)
    block = config.block_size

    def load(name):
        path = CLEAN / f"{name}.bin"
        return np.fromfile(path, dtype=np.uint16) if args.in_memory else np.memmap(path, dtype=np.uint16, mode="r")

    focus, general = load("focus_train"), load("general_train")
    val_sets = {k: np.memmap(CLEAN / f"{k}.bin", dtype=np.uint16, mode="r") for k in ("focus_val", "general_val")}
    print(f"focus_train {len(focus):,} | general_train {len(general):,} tokens; "
          + " | ".join(f"{k} {len(v):,}" for k, v in val_sets.items()))

    model = Model(config).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=args.peak_lr, betas=(0.9, 0.95), fused=True)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")
    OUT.mkdir(exist_ok=True)
    if ddp:
        dist.barrier()  # every rank checks for the resume checkpoint before rank 0 can write anything
    last_path, best_path, log_path = OUT / "pre350_last.pt", OUT / "pre350_best.pt", OUT / "pre350_log.csv"

    start_iter, best_focus, sampler_state = 0, float("inf"), {}
    if last_path.exists():
        ckpt = load_checkpoint(last_path, device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter, best_focus = ckpt["iter_num"] + 1, ckpt["best_val_loss"]
        sampler_state = ckpt.get("samplers", {})
        print(f"resumed from {last_path} at iter {start_iter}")
    else:
        init_weights(model, config)  # same seed on every rank; DDP also broadcasts rank 0's weights
        if master:
            log_path.write_text("iter,lr," + ",".join(val_sets) + "\n")

    samplers = {name: ChunkSampler(data, block, seed=args.seed + k, **sampler_state.get(name, {}))
                for k, (name, data) in enumerate([("focus", focus), ("general", general)])}
    raw_model = model  # saved without the "_orig_mod." prefix torch.compile adds
    if args.compile:
        model = torch.compile(model)
    eval_model = model  # rank 0 evaluates alone, outside DDP
    if ddp:
        model = DDP(model, device_ids=[torch.cuda.current_device()])

    def save(path, it):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "iter_num": it, "best_val_loss": best_focus, "args": vars(args), "config": asdict(config),
                    "samplers": {k: s.state() for k, s in samplers.items()}}, tmp)
        os.replace(tmp, path)  # atomic: a crash mid-save never corrupts the previous checkpoint

    rng = np.random.default_rng(args.seed + start_iter)
    tokens_per_iter = args.batch_size * args.grad_accum * block
    model.train()
    t0, last_print = time.time(), start_iter - 1
    for it in range(start_iter, args.max_iters + 1):
        lr = get_lr(it, args)
        if (it % args.eval_interval == 0 or it == args.max_iters) and master:
            losses = {k: evaluate(eval_model, v, args, block, device) for k, v in val_sets.items()}
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
        accum_loss = torch.zeros((), device=device)
        mine = range(rank, args.grad_accum, world)
        for micro in range(args.grad_accum):
            rows = get_mixed_rows(samplers, args, rng)  # every rank draws every micro-batch: samplers stay in sync
            if micro not in mine:
                continue
            x, y = to_device(rows, device)
            if ddp:
                model.require_backward_grad_sync = micro == mine[-1]  # all-reduce only after the last one
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, target=y)
            loss = loss / args.grad_accum
            accum_loss += loss.detach()
            (loss * world).backward()  # DDP averages over ranks; x world makes it the sum over all micro-batches
        if ddp:
            dist.all_reduce(accum_loss)
        accum_loss = accum_loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if it % 10 == 0:
            dt = time.time() - t0
            rate = tokens_per_iter * (it - last_print) / dt
            t0, last_print = time.time(), it
            print(f"iter {it} | loss {accum_loss:.4f} | lr {lr:.2e} | {rate:,.0f} tok/s", flush=True)
        if it % args.ckpt_interval == 0 and it > start_iter and master:
            save(last_path, it)

    if master:
        save(last_path, args.max_iters)
    print(f"done; best focus_val {best_focus:.4f}; "
          + " | ".join(f"{k} epoch {s.epoch} + {s.pos / s.n_chunks:.0%}" for k, s in samplers.items()))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
