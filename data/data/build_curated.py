"""
Build a curated continued-training corpus from fresh Institutional Books shards.

Per book (metadata only): ocr_score_src > 90, date1_src <= 1940, language_gen == "eng",
not a likely duplicate, LAW / AUXILIARY SCIENCES OF HISTORY kept with probability
--downsample-keep. Books go to the "focus" pool (philosophy/religion, language &
literature) or "general".

Output (uint16, <|endoftext|> after every book): {focus,general}_{train,val}.bin + stats.json

Usage:
    python build_curated.py --shards 1100-1199
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER = ROOT / "bpe" / "artifacts" / "institutional_books_ocr_gt_90_pre1940_50304" / "tokenizer.json"
DATASET_ID = "institutional/institutional-books-1.0"
FOCUS_TOPICS = {"PHILOSOPHY. PSYCHOLOGY. RELIGION", "LANGUAGE AND LITERATURE"}
DOWNSAMPLE_TOPICS = {"LAW", "AUXILIARY SCIENCES OF HISTORY"}
META_COLUMNS = ["barcode_src", "date1_src", "ocr_score_src", "language_gen",
                "topic_or_subject_gen", "likely_duplicates_barcodes_gen"]


def stable_fraction(text: str) -> float:
    """Deterministic pseudo-random number in [0, 1) from a string (reproducible sampling)."""
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) / 2**32


def load_shard(index: int, args, tokenizer: Tokenizer, eos_id: int) -> list[dict]:
    """Download one shard, filter books by metadata, tokenize the kept ones. Runs in a worker thread."""
    name = f"data/train-{index:05d}-of-09831.parquet"
    path = hf_hub_download(DATASET_ID, name, repo_type="dataset", local_dir=args.download_dir)
    pf = pq.ParquetFile(path)
    meta = pf.read(columns=META_COLUMNS).to_pylist()

    keep = []
    for row_idx, r in enumerate(meta):
        year = str(r["date1_src"] or "")
        if (r["ocr_score_src"] or 0) <= 90 or not year.isdigit() or int(year) > 1940:
            continue
        if r["language_gen"] != "eng":
            continue
        if r["topic_or_subject_gen"] in DOWNSAMPLE_TOPICS and stable_fraction(r["barcode_src"]) >= args.downsample_keep:
            continue
        keep.append((row_idx, r))
    if not keep:
        return []

    texts = pf.read(columns=["text_by_page_gen"]).column(0)
    books = []
    for row_idx, r in keep:
        # Same page joining as the original collector
        text = "\n\n".join(p.strip() for p in (texts[row_idx].as_py() or []) if p and p.strip())
        if text:
            books.append({"row": row_idx, "meta": r, "text": text})
    for book, enc in zip(books, tokenizer.encode_batch([b["text"] for b in books])):
        book["tokens"] = np.array(enc.ids + [eos_id], dtype=np.uint16)
        del book["text"]
    return books


def main():
    parser = argparse.ArgumentParser(description="Build the curated continued-training corpus.")
    parser.add_argument("--shards", default="1100-1199", help="Inclusive shard index range, e.g. 1100-1199")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed" / "curated")
    parser.add_argument("--download-dir", type=Path, default=ROOT / "data" / "raw" / "curated_parquet")
    parser.add_argument("--downsample-keep", type=float, default=0.25, help="Keep prob. for law/records books")
    parser.add_argument("--val-frac", type=float, default=0.01, help="Fraction of books held out for validation")
    parser.add_argument("--workers", type=int, default=4, help="Threads for download + tokenization")
    args = parser.parse_args()

    lo, hi = map(int, args.shards.split("-"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    eos_id = tokenizer.get_vocab()["<|endoftext|>"]

    keys = [f"{pool}_{split}" for pool in ("focus", "general") for split in ("train", "val")]
    files = {k: open(args.out_dir / f"{k}.bin", "wb") for k in keys}
    stats = {k: {"books": 0, "tokens": 0} for k in keys}
    stats["books_dup"] = 0
    seen = set()

    # Threads download/tokenize ahead; results arrive in shard order so dedup is deterministic
    with ThreadPoolExecutor(args.workers) as pool:
        shards = pool.map(lambda i: load_shard(i, args, tokenizer, eos_id), range(lo, hi + 1))
        for n, books in enumerate(shards, 1):
            for book in books:
                r = book["meta"]
                if r["barcode_src"] in seen or any(d in seen for d in (r["likely_duplicates_barcodes_gen"] or [])):
                    stats["books_dup"] += 1
                    continue
                seen.add(r["barcode_src"])
                pool_name = "focus" if r["topic_or_subject_gen"] in FOCUS_TOPICS else "general"
                split = "val" if stable_fraction("val" + r["barcode_src"]) < args.val_frac else "train"
                key = f"{pool_name}_{split}"
                book["tokens"].tofile(files[key])
                stats[key]["books"] += 1
                stats[key]["tokens"] += len(book["tokens"])
            print(f"[{n}/{hi - lo + 1}] shard {lo + n - 1}: focus_train {stats['focus_train']['tokens']:,} "
                  f"| general_train {stats['general_train']['tokens']:,} tokens", flush=True)

    for f in files.values():
        f.close()
    stats["args"] = {k: str(v) for k, v in vars(args).items()}
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "args"}, indent=2))


if __name__ == "__main__":
    main()
