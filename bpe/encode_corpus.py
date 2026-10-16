import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Encode txt shards into a single uint16 binary corpus.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "institutional_books_ocr_gt_90_txt",
        help="Directory containing txt shards.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=ROOT / "bpe" / "artifacts" / "institutional_books_ocr_gt_90_pre1940_50304" / "tokenizer.json",
        help="Path to tokenizer.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Directory to save encoded binary artifacts.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="institutional_books_ocr_gt_90_pre1940_50304",
        help="Base filename for encoded outputs.",
    )
    parser.add_argument(
        "--flush-every-files",
        type=int,
        default=1,
        help="Flush encoded token buffer to disk every N files.",
    )
    return parser.parse_args()


def flush_buffer(output_path: Path, buffer: list[int]) -> int:
    if not buffer:
        return 0
    arr = np.asarray(buffer, dtype=np.uint16)
    with output_path.open("ab") as f:
        arr.tofile(f)
    written = int(arr.size)
    buffer.clear()
    return written


def main():
    args = parse_args()

    txt_files = sorted(args.data_dir.glob("train-*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No txt shards found under {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.output_stem}.bin"
    metadata_path = args.output_dir / f"{args.output_stem}.json"

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    vocab = tokenizer.get_vocab()
    eos_id = vocab.get("<|endoftext|>")
    if eos_id is None:
        raise ValueError("Tokenizer is missing <|endoftext|> token")
    if len(vocab) > np.iinfo(np.uint16).max + 1:
        raise ValueError("Tokenizer vocab does not fit into uint16")

    if output_path.exists():
        output_path.unlink()

    total_tokens = 0
    buffer: list[int] = []

    print(f"Encoding {len(txt_files):,} txt shards with tokenizer {args.tokenizer_path}")
    print(f"Writing uint16 tokens to {output_path}")

    for idx, path in enumerate(txt_files, start=1):
        text = path.read_text(encoding="utf-8")
        encoded = tokenizer.encode(text)
        buffer.extend(encoded.ids)
        buffer.append(eos_id)

        if idx % args.flush_every_files == 0:
            total_tokens += flush_buffer(output_path, buffer)
            print(f"[{idx}/{len(txt_files)}] encoded {path.name}; total tokens so far: {total_tokens:,}", flush=True)

    total_tokens += flush_buffer(output_path, buffer)
    print(f"Finished encoding. Total tokens written: {total_tokens:,}", flush=True)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "data_dir": str(args.data_dir),
        "num_txt_files": len(txt_files),
        "tokenizer_path": str(args.tokenizer_path),
        "output_path": str(output_path),
        "dtype": "uint16",
        "total_tokens": total_tokens,
        "eos_token": "<|endoftext|>",
        "eos_id": eos_id,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
