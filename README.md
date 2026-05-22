# MiniLLM Architecture — Building an LLM from Scratch

> A step-by-step journey from a single learnable weight all the way to a working Transformer-based language model, written entirely from scratch using PyTorch.

This repo documents a personal learning project: understanding how large language models actually work — not by reading about them, but by building every piece by hand, one concept at a time.

---

## The Philosophy

Every file in this repo adds exactly one idea on top of the last. No magic. No black boxes. By the time you reach `FINAL.py`, you've touched gradient descent, neurons, layers, activation functions, the `nn.Module` abstraction, token embeddings, raw self-attention, a proper Transformer block, and a tokenizer — all written and understood from the ground up.

---

## The Journey

### `step1.py` — A Single Learnable Weight

The very beginning. One input, one weight, one target. Gradient descent loop written by hand using `loss.backward()` and manual `w -= lr * w.grad`. The goal was simple: understand what it even means for a weight to *learn*.

```python
x = torch.tensor(2.0)
w = torch.tensor(1.5, requires_grad=True)
# ... gradient descent loop
```

---

### `step2.py` — Multiple Inputs, One Output (Dot Product)

Extended to a vector of inputs and a vector of weights. Introduced `torch.dot` to compute a weighted sum — essentially one neuron computing a scalar output from a 3D input. The weights still learn via manual gradient updates.

---

### `step3.py` — One Layer, Multiple Outputs

Upgraded to a weight matrix `W` (2×3) and a bias vector, computing `y = W @ x + b`. This is a full linear layer — two neurons reading the same three inputs and producing two separate outputs. First time using matrix multiplication for neural computation.

---

### `step4.py` — Two Layers: A Real Neural Network

Stacked two linear layers manually (`w1`, `b1` → `w2`, `b2`), with the output of the first feeding into the second. Both sets of weights are updated independently via backprop. This is the first proper multi-layer network — a tiny feedforward MLP built from scratch.

---

### `step5.py` — Activation Functions (ReLU)

Added `F.relu()` after the first layer. Without an activation function, stacking linear layers is mathematically equivalent to one linear layer — activations are what give networks the power to model nonlinear relationships. This step made the network genuinely deeper.

---

### `step6.py` — Wrapping It in `nn.Module`

Rewrote the two-layer network as a proper PyTorch `nn.Module` class (`SimpleNetwork`). Replaced manual gradient zeroing and weight updates with `torch.optim.SGD`. This is the standard PyTorch pattern that all real models use — the code became cleaner and the model was now a reusable object.

```python
class Simplenetwork(nn.Module):
    def __init__(self):
        self.layer1 = nn.Linear(3, 2)
        self.layer2 = nn.Linear(2, 1)
```

---

### `step7.py` — Token Embeddings

Shifted focus toward language. Introduced `nn.Embedding` — a lookup table that maps integer token IDs to dense float vectors. This is how LLMs represent words: each token ID maps to a learned vector in a high-dimensional space. Looked up single tokens and batches of tokens.

```python
embedding = nn.Embedding(10, 4)  # 10 possible words, each as a 4D vector
```

---

### `step8.py` — Raw Self-Attention (From Scratch)

Implemented the core Transformer operation by hand, without any module wrapping. Computed Query, Key, and Value matrices by multiplying token embeddings against learned weight matrices `Wq`, `Wk`, `Wv`. Then computed attention scores via `Q @ K.T`, applied softmax to get attention weights, and combined with `V` to get the attended output.

This is the heart of every modern LLM.

```python
scores = Q @ K.T
weights = F.softmax(scores, dim=-1)
output = weights @ V
```

---

### `step9.py` — Self-Attention as an `nn.Module` + LayerNorm

Wrapped the self-attention logic into a clean `SelfAttention` class. Added a residual connection (`output + x`) and `nn.LayerNorm` to stabilize training — two critical components from the original "Attention Is All You Need" paper. The architecture now properly follows the Transformer block pattern.

---

### `LLM.py` — A Complete Tiny Transformer

Combined everything into `TinyLLM`: a proper Transformer-style language model with:

- Token embeddings (`nn.Embedding`)
- Multiple stacked Transformer blocks (each with self-attention + a feedforward sublayer)
- Scaled dot-product attention (`/ sqrt(embed_dim)`) to prevent softmax saturation
- A final linear projection over the vocabulary (`vocab_size = 1000`)

```python
class TinyLLM(nn.Module):
    def __init__(self, embed_dim, num_blocks):
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([SelfAttention(embed_dim) for _ in range(num_blocks)])
        self.final_linear = nn.Linear(embed_dim, vocab_size)
```

Feed it a sequence of token IDs → get back logits over the vocabulary. That's a language model.

---

### `tokenizer.py` + `LLM+tokenizer.py` — Bringing Text In

Built a tokenizer to convert raw text into token IDs (and back), then connected it to the model pipeline. With this, the system can take a string of text, tokenize it, pass it through the Transformer, and produce output logits — the full end-to-end loop of a language model.

---

### `FINAL.py` — Everything Together

The complete, integrated implementation: tokenizer + embeddings + stacked Transformer blocks + output projection, all wired together in one clean file. The culmination of every concept introduced across steps 1–9.

---

## What Was Built

```
Raw gradient descent
    → Neurons
        → Linear layers
            → Multi-layer networks
                → Activation functions (ReLU)
                    → nn.Module abstraction
                        → Token embeddings
                            → Self-attention
                                → Transformer blocks
                                    → Full LLM
```

---

## Tech Stack

- **Python**
- **PyTorch** (`torch`, `torch.nn`, `torch.nn.functional`, `torch.optim`)

---

## How to Run

Each step is a standalone script. Run any of them with:

```bash
python step1.py   # or step2.py, step3.py ... LLM.py, FINAL.py
```

No external datasets or pretrained weights needed — all examples use random or hardcoded tensors.

---

## Why This Exists

Most tutorials hand you a finished model and ask you to tweak hyperparameters. This project does the opposite: it starts from the most primitive possible thing (one number that learns) and builds up every abstraction by hand. If you've ever wanted to truly understand what's happening inside an LLM rather than just use one, this is that journey.

---

*Built by [Sujay](https://github.com/Sujay510) — CSE @ IIT Ropar, 2024–2028*
