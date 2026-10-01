# Vintage-LLM: a GPT-style model pretrained from scratch on pre-1940 books

A 124M-parameter decoder-only Transformer written in plain PyTorch, with no `transformers` model classes. It is pretrained on ~1.6B tokens of English books published before 1940, so it writes like old books do.

## Highlights

- **Model:** 12 layers, 12 heads, 768-d, 1024-token context. Uses RoPE, SwiGLU, RMSNorm, FlashAttention (`F.scaled_dot_product_attention`), and weight-tied embeddings. 123.6M unique parameters.
- **Tokenizer:** a byte-level BPE written from scratch (`bpe/base.py`) with a 50,304-token vocabulary. Encoding runs in parallel over 8 worker processes into a `uint16` memory-mapped file (`bpe/main.py`).
- **Data:** the Hugging Face *Institutional Books* corpus, filtered to OCR quality > 90% and publication year before 1940. That leaves 1,089 text shards and ~1.59B tokens.
- **Training:** bf16 autocast, gradient accumulation (4 × 8 × 1024 = 32,768 tokens per step), gradient clipping, AdamW with cosine warmup/decay, validation-based best-checkpoint saving, and full resume from model and optimizer state.

## Results so far

- Trained for 12,000 steps (~393M tokens) on a single NVIDIA A100. Best validation loss is 4.60.
- Sample (prompt in **bold**, temperature 0.8, top-k 40):

  > **The railway had at last reached the village, and** the attack of the enemy on the morning of the 4th of April. It is now ready to take the field and take fire on the side of the enemy. The enemy were here engaged in a position so to go out and take fire…

The run is a partial one: it is 12K steps into a 100K-step schedule, so the text is fluent in period style but not yet coherent over long spans.

## Repository layout

```
bpe/base.py     BPE tokenizer: train / encode / decode / save / load
bpe/main.py     parallel corpus encoding -> uint16 memmap
model/model.py  model definition + training loop (python model/model.py)
```

## Run

```bash
pip install torch numpy python-dotenv
# 1. train the tokenizer and encode the corpus (edit paths in bpe/main.py)
python bpe/main.py
# 2. train; DATA_DIR points to the encoded .bin file
DATA_DIR=./data/books_data_final.bin OUT_DIR=./out python model/model.py
```

Training resumes automatically from `OUT_DIR/last_model.pt` if that file exists.
