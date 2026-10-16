# Vintage-LLM: a GPT-style model pretrained from scratch on pre-1940 books

A decoder-only Transformer written in plain PyTorch (no `transformers` model classes), pretrained only on English books published before 1940. It has never seen the internet, so it writes like an old book: give it the start of a maxim and it finishes the thought in period style.

```
Knowledge is a matter of the heart, and not of the head.
Prudence is the mother of the virtues, and vice is the parent of vice.
Kindness is the first law of nature.
Duty is the same, whether it be performed in the day or the night.
```

## Models

| Model | Shape | Params | Training data | focus_val | general_val |
|---|---|---|---|---|---|
| 124M | 12 layers x 768 | 123.6M | 1.6B tokens, raw (~36% non-English) | 3.695 | 3.616 |
| 124M + anneal | same | 123.6M | + continued training on curated English | 3.658 | 3.550 |
| **350M** | **24 layers x 1024** | **359.8M** | **8.4B tokens, page-cleaned English** | **2.868** | **2.721** |

All three are scored on the same held-out batches (cross-entropy in nats, lower is better). *focus* = philosophy, religion, literature and essay-like books; *general* = everything else. On focus text the 350M model's perplexity is about 18, versus about 39 for the annealed 124M.

Architecture (both sizes): RoPE, SwiGLU, RMSNorm, FlashAttention via `F.scaled_dot_product_attention`, weight-tied embeddings, 1024-token context, 50,304-token byte-level BPE vocabulary.

## Data pipeline

Source: the Hugging Face [Institutional Books](https://huggingface.co/datasets/institutional/institutional-books-1.0) corpus. The first model was trained on raw text dumps; an audit found about a third of it was German, French, Latin or Greek, plus index pages and tables. The 350M model uses a cleaned corpus built by `data/data/build_clean.py`:

- **Book filters:** OCR score > 90, published 1940 or earlier, English, deduplicated; law and archival-records books down-sampled to 25%.
- **Page filters:** each page must be confidently English (fastText `lid.176`, p >= 0.65), at least 300 characters, mostly letters, not an index or table (digit density and short numbered lines), and not library boilerplate.
- **Pools:** books are split into a *focus* pool (philosophy/religion, literature, or titles like *Essays*, *Maxims*, *Sermons*, *Letters*) and a *general* pool. Training samples 45% focus, 55% general.

Result: 40,629 books, 8.71B training tokens (3.18B focus, 5.53B general), about 84% of characters kept.

## Training the 350M model

`model/pretrain.py`: AdamW (betas 0.9/0.95, weight decay 0.1), peak LR 3e-4 with 700 warmup steps and cosine decay to 3e-5, 0.5M tokens per step, 16,000 steps (8.4B tokens), bf16, `torch.compile`. Data is read in a shuffled order of 1024-token chunks so nothing repeats within an epoch; the sampler position is checkpointed, so a resumed run continues on unseen data.

It runs on one GPU or on several with `torchrun`. In multi-GPU mode every rank draws the same micro-batches and computes only its share, so the result is the same as on one GPU (verified: identical losses on 1 vs 2 GPUs). The run started on a single B200 (263k tokens/s) and was finished on 8x H100 (1.24M tokens/s).

## Generating text

`model/inference.py` samples several candidate sentences (temperature 0.8, top-k 30, no repeated 3-token sequences), keeps only those that finish within 12 words, drops any that repeat a phrase, and prints the one the model scores most fluent. Short answers stay on topic; long ones tend to drift into whatever the word meant most often in old books.

```bash
cd model
python inference.py --ckpt out/pre350_best.pt --prompt "Freedom is"
# Freedom is the gift of God.
```

Useful flags: `--max-words 0` (no length limit), `--frame` (prepend hidden maxims to steer toward aphorisms), `--verbose` (print every candidate with its score). See [`model/samples.md`](model/samples.md) for all 19 test prompts, misses included: *Fortune* is still often a place name, and *Justice* a judge.

## Repository layout

```
bpe/
  base.py, main.py        original from-scratch BPE tokenizer + parallel encoder
  train_bpe.py            the tokenizer actually used (Hugging Face tokenizers, 50,304 vocab)
  encode_corpus.py        txt shards -> uint16 token file
  artifacts/.../tokenizer.json
data/data/
  build_clean.py          page-cleaned corpus (used for the 350M model)
  build_curated.py        curated corpus used for annealing the 124M model
model/
  train.py                model definition + the original 124M training loop
  pretrain.py             350M pretraining (single- or multi-GPU)
  continue_train.py       annealing / continued training; shared sampling helpers
  inference.py            sentence generation
  samples.md              sample outputs
pod_pretrain.sh           cloud launcher: download data, train, push checkpoints
```

Data files and checkpoints are not in the repo (`data/processed/`, `model/out/`).

## Run

```bash
pip install torch numpy tokenizers pyarrow huggingface_hub fasttext python-dotenv
# 1. build the cleaned corpus (downloads shards; needs data/models/lid.176.bin from fastText)
python data/data/build_clean.py --shards 1100-1199,1200-9830:8
# 2. pretrain (resumes automatically from model/out/pre350_last.pt)
cd model
python pretrain.py --compile                                # one GPU
torchrun --standalone --nproc_per_node 8 pretrain.py --compile   # eight GPUs
```
