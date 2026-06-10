# MiniLLM Architecture

A Transformer-based language model built from scratch in PyTorch  no HuggingFace, no high-level LLM abstractions. Every component is implemented manually to understand what actually happens inside a language model.

---

## What This Is

This project builds a working LLM pipeline across incremental steps, starting from raw gradient descent and ending at a model that tokenizes text, trains on next-token prediction, and autoregressively generates output.

The goal was not to train a useful model  it was to understand the mechanics of every component by implementing them from first principles.

---

---

## Components (by file)

### `tokenizer.py` : Word-level tokenizer
Splits text on whitespace, builds a sorted vocabulary, maps words to integer IDs and back. No OOV handling — toy use only.

### `simplenetwork.py` : Baseline feedforward network
A 2-layer MLP with ReLU trained via SGD to verify the gradient descent loop before adding any Transformer complexity.

### `selfattention.py` : Self-attention module
Implements Q/K/V projections, scaled dot-product attention (`scores / √embed_dim`), and LayerNorm with residual connection.

The scaling by `√embed_dim` prevents softmax saturation: without it, large dot products push the softmax toward a one-hot distribution, causing vanishing gradients and stalled training.

### `llm.py` : Full TinyLLM

Combines all components into a working model:

**Causal mask** : prevents each token from attending to future positions. Implemented as an upper-triangular mask filled with `-inf` before softmax, so future positions contribute exactly zero attention weight (`e^-inf = 0`).

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
```

**Positional embeddings** : learned embeddings added to token embeddings so the model can distinguish token order. Addition (not concatenation) keeps the hidden dimension constant across all downstream layers.

```python
h = self.token_emb(x) + self.pos_emb(positions)
```

**Feed-forward network** : inner dimension is 4× the embedding dimension, following the original Transformer paper. Applied per-token after attention.

**Training** : next-token prediction with cross-entropy loss. Input is `ids[:-1]`, target is `ids[1:]`.

**Generation** : autoregressive: feed the current sequence, take the last token's logits, argmax to get the next token, append and repeat.

---

## Training Results

```
Corpus     : "hello i am sujay you are"
Vocab size : 6
embed_dim  : 8
num_blocks : 2
Optimizer  : Adam, lr=0.01
```

| Iteration | Loss   |
|-----------|--------|
| 0         | 2.0809 |
| 100       | 0.0091 |
| 500       | 0.0006 |
| 1000      | 0.0002 |

The model memorizes the 6-word corpus, which is expected  the point was to verify the pipeline works end-to-end, not to generalize.

```
Generated: "hello i am sujay you are"
```

---

## Key Concepts Implemented from Scratch

| Concept | Where |
|---|---|
| Gradient descent + backprop | `simplenetwork.py` |
| Word-level tokenization | `tokenizer.py` |
| Q/K/V attention + scaling | `selfattention.py`, `llm.py` |
| Causal masking | `llm.py` |
| Residual connections | `llm.py` |
| LayerNorm | `llm.py` |
| Positional embeddings | `llm.py` |
| Next-token prediction training | `llm.py` |
| Autoregressive generation | `llm.py` |

---

## Known Limitations

- Word-level tokenizer with no OOV handling, words outside the training vocabulary will crash encode
- No multi-head attention, single head only
- FFN is inside the SelfAttention class rather than a separate TransformerBlock, architecturally merged for simplicity
- Trained on a 6-word corpus, the model memorizes rather than generalizes
- Generation uses argmax (greedy), deterministic, no temperature sampling

---

## Requirements

```
torch
```

```bash
pip install torch
python llm.py
```
