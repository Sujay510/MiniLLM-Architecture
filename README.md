# MiniLLM Architecture

A Transformer-based language model built from scratch in PyTorch. No HuggingFace, no high-level LLM abstractions. Every component is implemented manually to understand what actually happens inside a language model.

---

## What This Is

This project builds a working LLM pipeline across incremental steps, starting from raw gradient descent (a single weight learning `w * x = y`) and ending at a full GPT-2 style transformer that trains on Shakespeare and generates text autoregressively.

The goal was never to build something competitive with a real model. It was to understand every piece by building it, breaking it, and fixing it.

---

## Current Architecture (FINAL.py)

| Parameter | Value |
|-----------|-------|
| `embed_dim` | 64 |
| `num_blocks` | 4 |
| `num_heads` | 8 |
| `ffn_dim` | 256 (4x embed_dim) |
| `vocab_size` | ~745 (BPE, trained on Shakespeare corpus) |
| `block_size` | 32 |
| `batch_size` | 16 |
| `dropout` | 0.1 |
| `optimizer` | Adam, warmup + cosine decay (max_lr 5e-4, min_lr 5e-5) |
| `loss` | Cross-entropy (next-token prediction) |
| `device` | MPS (Apple Silicon) with CPU fallback |

---

## Components

### `bytepairencoder.py` - BPE tokenizer
Trains subword merges on the corpus and encodes/decodes text into subword tokens. This replaced the earlier word-level tokenizer, which crashed on any word it hadn't seen during training. BPE never crashes on unknown input since it can always fall back to smaller pieces.

### `TinyLLM` (in FINAL.py) - the model
Token embeddings + learned positional embeddings, stacked `TransformerBlock`s, and a final linear layer projecting to vocab size.

### `MultiHeadAttention`
Splits Q, K, V into multiple heads so different heads can specialize in different kinds of relationships between tokens (grammar, position, semantics, etc.), then concatenates and projects back with `Wo`. Uses scaled dot-product attention (`scores / sqrt(head_dim)`) with a causal mask so tokens can't attend to future positions.

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
scores = scores.masked_fill(mask, float('-inf'))
```

### `TransformerBlock`
Pre-LayerNorm architecture (GPT-2 style), not the original post-LN from the "Attention is All You Need" paper:

```python
output = self.attention(self.norm1(x))
x = self.dropout(output) + x
ff_output = self.ff(self.norm2(x))
x = self.dropout(ff_output) + x
```

Normalizing before the sublayer instead of after made training noticeably more stable and let the model converge faster once we switched.

### Dropout
Added inside `MultiHeadAttention` (after the output projection) and inside `TransformerBlock` (before both residual connections). Dropout has no learnable parameters, so switching between architectures without it and with it is safe as long as the rest of the model didn't change.

### Training loop
Batches are sampled randomly from the corpus rather than read sequentially, so the model can't just memorize word order. Train/val split is 90/10, and the validation batch is fixed once at the start of training rather than resampled every check, so loss readings are comparable across iterations instead of just noise.

Learning rate follows linear warmup into cosine decay:

```python
def getlr(current, warmup, total, max_lr, min_lr):
    if current < warmup:
        return max_lr * (current / warmup)
    progress = (current - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

Checkpointing saves both model and optimizer state, since Adam's momentum and velocity estimates are just as important to preserve as the weights themselves. Resuming without the optimizer state means losing all that accumulated momentum and effectively restarting the optimizer from scratch.

### Generation
Uses top-k sampling with temperature instead of greedy argmax, which was getting stuck in repetition loops (`I have not not not not revenge`). `k=10, temperature=1.0` gave a decent balance between coherence and variety for this model size.

```python
def top_k_sample(logits, k, temperature=1.0):
    ...
    sample = torch.multinomial(probs, num_samples=1)
    return indices[sample].item()
```

### GPU support
Runs on Apple Silicon via MPS. Every tensor that touches the model (batches, causal masks, position indices, generation input) has to be moved to the same device as the model, or PyTorch throws a device mismatch error. The device is read from `x.device` inside modules rather than a hardcoded global, so the code isn't tied to one specific backend.

---

## Training Results

Trained on the full Shakespeare corpus, 50k+ iterations across several sessions (checkpointed and resumed each time).

```
Iter 0:     train_loss=6.9523, val_loss=6.9657   (fresh start after Pre-LN switch)
Iter 40000: train_loss=0.2403, val_loss=0.2928
Iter 49000: train_loss=0.3351, val_loss=0.3677   (after adding dropout + LR schedule)
```

Sample generation:

```
Prompt: "My name is Sujay"
Output: "...ROMEO: Gither the fear the worn in the ladence,"
```

Not coherent English, but the model reliably reproduces Shakespeare-style formatting (character names in caps followed by a colon, period-appropriate contractions like "corn'd", archaic phrasing) which shows it actually learned structural patterns from the corpus rather than just memorizing.

---

## What Changed From the Toy Version

The very first version of this model (single-head attention, word-level tokenizer, 6-word vocab, post-LN, no dropout, argmax generation) is still a good reference for understanding the basics without the added complexity. The current version adds:

- BPE tokenizer instead of word-level (no more crashing on unseen words)
- Multi-head attention instead of single-head
- Separate `TransformerBlock` class instead of FFN merged into the attention module
- Pre-LayerNorm instead of post-LayerNorm
- Dropout regularization
- Fixed validation batch instead of resampling every check
- Batched training on a real corpus instead of one sentence
- GPU (MPS) support
- Learning rate warmup + cosine decay instead of a constant LR
- Top-k + temperature sampling instead of greedy argmax
- Checkpointing for both model and optimizer state

---

## Known Limitations

- Word-level BPE vocab is still corpus-specific (~745 tokens), nowhere near GPT-2's 50k
- No weight decay yet (plain Adam, not AdamW)
- No gradient clipping
- Single machine, single GPU, no distributed training
- The model has only ever seen Shakespeare, so it has no general world knowledge and can't answer questions, only continue Shakespeare-style text

---

## What's Next

- [ ] Weight decay (AdamW)
- [ ] Gradient clipping
- [ ] Retrieval-augmented generation (RAG): embed a small document set, retrieve relevant context by similarity, feed it to the model alongside the question
- [ ] Train on structured Context/Question/Answer data so the model can actually learn to use retrieved context, not just continue Shakespeare

---

## Requirements

```
torch
```

## Setup

These three files need to be in the same folder for `FINAL.py` to run:

```
FINAL.py
bytepairencoder.py
shakespeare.txt
```

`FINAL.py` trains the BPE tokenizer directly on `shakespeare.txt` at the top of the script, and imports `bytepairencoder.py` for the actual encode/decode/train_bpe logic. If either file is missing or named differently, it'll fail before training even starts.

The first run will also create `model.pt` and `optimizer.pt` in the same folder once training finishes. These get loaded automatically on later runs to resume training, so delete them if you want to start fresh (or if you change the architecture, see Known Limitations).

```bash
pip install torch
python FINAL.py
```