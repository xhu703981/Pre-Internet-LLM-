import argparse
import json
from datetime import datetime
from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, processors
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer from local txt files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "institutional_books_ocr_gt_90_txt",
        help="Directory containing .txt shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "bpe" / "artifacts" / "institutional_books_ocr_gt_90_pre1940_50304",
        help="Directory to save tokenizer artifacts.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50304,
        help="Target tokenizer vocabulary size.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency to keep during BPE training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    txt_files = sorted(args.data_dir.glob("train-*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No training txt files found under {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.output_dir / "tokenizer.json"
    metadata_path = args.output_dir / "metadata.json"

    print(f"Found {len(txt_files):,} txt shards")
    print(f"Training byte-level BPE with vocab_size={args.vocab_size:,} min_frequency={args.min_frequency}")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<unk>", "<|endoftext|>"]
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train([str(p) for p in txt_files], trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save(str(tokenizer_path))

    metadata = {
        "created_at": datetime.now().isoformat(),
        "data_dir": str(args.data_dir),
        "num_txt_files": len(txt_files),
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "special_tokens": special_tokens,
        "tokenizer_path": str(tokenizer_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved tokenizer to {tokenizer_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
