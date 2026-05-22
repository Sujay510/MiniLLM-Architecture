# MiniLLM Architecture

Building a language model from scratch using PyTorch. No shortcuts, no copying — just building each piece one step at a time to actually understand how it works.

# Why

I wanted to understand what actually happens inside an LLM, not just use one. So I started from the most basic thing — a single weight that learns — and kept adding one concept at a time until I had something that looks like a real Transformer.

# The Steps

### step1.py — Single weight, gradient descent

One input, one weight, one target. The gradient descent loop is written by hand.

```python
x = torch.tensor(2.0)
w = torch.tensor(1.5, requires_grad=True)
# w learns to make w*x = 4.0
```

### step2.py — Multiple inputs, dot product

Extended to a vector of inputs. Uses `torch.dot` to compute a weighted sum — basically one neuron.

### step3.py — Weight matrix, multiple outputs

Now `W` is a 2x3 matrix. Two neurons reading the same three inputs, producing two separate outputs. First time using matrix multiplication.

```python
y = w @ x + b
```

### step4.py — Two layers stacked

Two linear layers, both learning at the same time. The output of layer 1 feeds into layer 2. First real multi-layer network.

### step5.py — ReLU activation

Added `F.relu()` after the first layer. Without an activation, stacking linear layers is still just one linear layer mathematically. This is what makes it actually deeper.

### step6.py — nn.Module

Rewrote everything as a proper PyTorch class. Replaced manual gradient zeroing with `torch.optim.SGD`. This is the standard pattern all real models use.

```python
class SimpleNetwork(nn.Module):
    def __init__(self):
        self.layer1 = nn.Linear(3, 2)
        self.layer2 = nn.Linear(2, 1)
```

### step7.py — Token Embeddings

Shifted to language. `nn.Embedding` maps integer token IDs to float vectors. This is how LLMs represent words.

```python
embedding = nn.Embedding(10, 4)  # 10 words, each as a 4D vector
```

### step8.py — Self-attention from scratch

Implemented attention by hand without any module wrapper. Q, K, V matrices computed manually, scores via `Q @ K.T`, softmax, then weighted sum with V.

```python
scores = Q @ K.T
weights = F.softmax(scores, dim=-1)
output = weights @ V
```

### step9.py — Self-attention as nn.Module + LayerNorm

Wrapped the attention logic into a class. Added residual connection and LayerNorm — both from the original Transformer paper.

### LLM.py — Putting it all together

Combined embeddings + stacked attention blocks + feedforward sublayer + final linear projection into `TinyLLM`. Feed it token IDs, get back logits over the vocabulary.

```python
x = torch.tensor([1, 3, 7])  # 3 token ids
output = model(x)
print(output.shape)  # (3, 1000)
```

### tokenizer.py + LLM+tokenizer.py

Built a tokenizer to convert raw text to token IDs and back, then connected it to the model. This completes the full text-in to logits-out pipeline.

### FINAL.py

Everything in one clean file.

# What's missing

The model has random weights — it hasn't been trained on any real data yet. It also doesn't have causal masking (tokens shouldn't see future tokens during generation) or a proper text generation loop. Those are the next steps.

# Requirements

```
Python 3.x
PyTorch
```

# How to run

Each step is standalone, just run:

```bash
python step1.py
```

No datasets or pretrained weights needed — everything uses random or hardcoded tensors.
