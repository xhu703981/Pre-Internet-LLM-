"""
Build the from-scratch pretraining corpus with page-level cleaning.

Per book (metadata): ocr_score_src > 90, date1_src <= 1940, language_gen == "eng", not a duplicate,
LAW / AUXILIARY SCIENCES OF HISTORY kept with probability --downsample-keep.
Per page: dropped if short, library boilerplate, index/table-like, symbol garbage,
or not confidently English (fastText lid.176).
Pools: "focus" (philosophy/religion, literature, or a reflective title) vs "general".

Output (uint16, <|endoftext|> after every book): {focus,general}_{train,val}.bin + stats.json.
Progress is saved after every shard; rerunning the same command resumes.

Usage:
    python build_clean.py --shards 1100-1199,1200-9830:8
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from build_curated import DATASET_ID, DEFAULT_TOKENIZER, DOWNSAMPLE_TOPICS, FOCUS_TOPICS, ROOT, stable_fraction

LID_MODEL = ROOT / "data" / "models" / "lid.176.bin"
META_COLUMNS = ["barcode_src", "title_src", "date1_src", "ocr_score_src", "language_gen",
                "topic_or_subject_gen", "likely_duplicates_barcodes_gen"]
FOCUS_TITLE = re.compile(r"\b(essays?|thoughts|maxims|reflections|sermons|letters|discourses?|"
                         r"meditations|aphorisms|proverbs)\b", re.I)
BOILERPLATE = re.compile(r"borrower must return|overdue (fines|notices)|widener library|digitized by|"
                         r"harvard college library|this book is due|date due", re.I)

_worker = {}


def init_worker():
    os.environ["RAYON_NUM_THREADS"] = "4"  # tokenizers' thread pool, per process
    import fasttext
    from tokenizers import Tokenizer
    _worker["lid"] = fasttext.load_model(str(LID_MODEL))
    _worker["tok"] = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    _worker["eos"] = _worker["tok"].get_vocab()["<|endoftext|>"]


def page_reason(text: str) -> str:
    """'keep' or the reason a page is dropped."""
    if len(text) < 300:
        return "short"
    if BOILERPLATE.search(text):
        return "boilerplate"
    chars = [c for c in text if not c.isspace()]
    digits = sum(c.isdigit() for c in chars) / len(chars)
    alpha = sum(c.isalpha() for c in chars) / len(chars)
    lines = [l for l in text.split("\n") if l.strip()]
    # index/table rows are short lines carrying numbers; prose paragraphs with a date are long lines
    numeric_short = sum(len(l) < 80 and bool(re.search(r"\d", l)) for l in lines) / len(lines)
    if digits > 0.08 or (numeric_short > 0.4 and len(lines) >= 5):
        return "numeric"
    if alpha < 0.75:
        return "garbage"
    # call the C++ predictor directly: the Python wrapper breaks under NumPy 2
    (prob, label), = _worker["lid"].f.predict(text.replace("\n", " "), 1, 0.0, "strict")
    if label != "__label__en" or prob < 0.65:
        return "language"
    return "keep"


def load_shard(index: int, download_dir: str, downsample_keep: float, delete: bool):
    """Download one shard, filter books and pages, tokenize. Runs in a worker process."""
    path = hf_hub_download(DATASET_ID, f"data/train-{index:05d}-of-09831.parquet",
                           repo_type="dataset", local_dir=download_dir)
    pf = pq.ParquetFile(path)
    meta = pf.read(columns=META_COLUMNS).to_pylist()
    keep = []
    for row_idx, r in enumerate(meta):
        year = str(r["date1_src"] or "")
        if (r["ocr_score_src"] or 0) <= 90 or not year.isdigit() or int(year) > 1940:
            continue
        if r["language_gen"] != "eng":
            continue
        if r["topic_or_subject_gen"] in DOWNSAMPLE_TOPICS and stable_fraction(r["barcode_src"]) >= downsample_keep:
            continue
        keep.append((row_idx, r))

    dropped = {}
    books = []
    if keep:
        texts = pf.read(columns=["text_by_page_gen"]).column(0)
        for row_idx, r in keep:
            pages = []
            for p in texts[row_idx].as_py() or []:
                p = (p or "").strip()
                if not p:
                    continue
                why = page_reason(p)
                if why == "keep":
                    pages.append(p)
                else:
                    dropped[why] = dropped.get(why, 0) + len(p)
            if pages:
                books.append({"meta": r, "text": "\n\n".join(pages), "chars": sum(map(len, pages))})
        encs = _worker["tok"].encode_batch([b["text"] for b in books])
        for book, enc in zip(books, encs):
            book["tokens"] = np.array(enc.ids + [_worker["eos"]], dtype=np.uint16)
            del book["text"]
    del pf
    if delete:
        os.remove(path)
    return index, books, dropped


def parse_shards(spec: str) -> list[int]:
    """'1100-1199,1200-9830:8' -> inclusive ranges with optional step."""
    out = []
    for part in spec.split(","):
        rng, _, step = part.partition(":")
        lo, hi = map(int, rng.split("-"))
        out.extend(range(lo, hi + 1, int(step or 1)))
    return out


def main():
    parser = argparse.ArgumentParser(description="Build the page-cleaned pretraining corpus.")
    parser.add_argument("--shards", default="1100-1199,1200-9830:8")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed" / "clean")
    parser.add_argument("--download-dir", type=Path, default=ROOT / "data" / "raw" / "clean_parquet_tmp")
    parser.add_argument("--downsample-keep", type=float, default=0.25, help="Keep prob. for law/records books")
    parser.add_argument("--val-frac", type=float, default=0.01, help="Fraction of books held out for validation")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--keep-parquet", action="store_true", help="Don't delete shards after processing")
    args = parser.parse_args()

    shards = parse_shards(args.shards)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_dir / "progress.json"
    keys = [f"{pool}_{split}" for pool in ("focus", "general") for split in ("train", "val")]

    if progress_path.exists():  # resume: cut files back to the last completed shard
        state = json.loads(progress_path.read_text())
        for k in keys:
            with open(args.out_dir / f"{k}.bin", "r+b") as f:
                f.truncate(state["offsets"][k])
        print(f"resuming after {state['done']} of {len(shards)} shards")
    else:
        state = {"done": 0, "offsets": {k: 0 for k in keys}, "seen": [],
                 "stats": {**{k: {"books": 0, "tokens": 0} for k in keys}, "books_dup": 0,
                           "chars_kept": 0, "chars_dropped": {}}}
        for k in keys:
            (args.out_dir / f"{k}.bin").write_bytes(b"")
    stats, seen = state["stats"], set(state["seen"])
    files = {k: open(args.out_dir / f"{k}.bin", "ab") for k in keys}

    todo = shards[state["done"]:]
    with ProcessPoolExecutor(args.workers, initializer=init_worker) as pool:
        # map yields in shard order, so dedup and output order are deterministic
        results = pool.map(load_shard, todo, [str(args.download_dir)] * len(todo),
                           [args.downsample_keep] * len(todo), [not args.keep_parquet] * len(todo))
        for index, books, dropped in results:
            for why, n in dropped.items():
                stats["chars_dropped"][why] = stats["chars_dropped"].get(why, 0) + n
            for book in books:
                r = book["meta"]
                if r["barcode_src"] in seen or any(d in seen for d in (r["likely_duplicates_barcodes_gen"] or [])):
                    stats["books_dup"] += 1
                    continue
                seen.add(r["barcode_src"])
                focus = r["topic_or_subject_gen"] in FOCUS_TOPICS or FOCUS_TITLE.search(r["title_src"] or "")
                split = "val" if stable_fraction("val" + r["barcode_src"]) < args.val_frac else "train"
                key = f"{'focus' if focus else 'general'}_{split}"
                book["tokens"].tofile(files[key])
                stats[key]["books"] += 1
                stats[key]["tokens"] += len(book["tokens"])
                stats["chars_kept"] += book["chars"]
            for f in files.values():
                f.flush()
            state["done"] += 1
            state["offsets"] = {k: f.tell() for k, f in files.items()}
            state["seen"] = sorted(seen)
            tmp = progress_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state))
            os.replace(tmp, progress_path)
            total = sum(stats[k]["tokens"] for k in keys)
            print(f"[{state['done']}/{len(shards)}] shard {index}: total {total / 1e9:.2f}B tokens "
                  f"(focus_train {stats['focus_train']['tokens'] / 1e9:.2f}B)", flush=True)

    for f in files.values():
        f.close()
    stats["args"] = {k: str(v) for k, v in vars(args).items()}
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "args"}, indent=2))


if __name__ == "__main__":
    main()
