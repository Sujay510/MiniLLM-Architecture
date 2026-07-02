import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Tokenizer
# ─────────────────────────────────────────────
class SimpleTokenizer:
    """Word-level tokenizer (toy use only — no OOV handling)."""

    def __init__(self, text: str):
        words = text.split()
        vocab = sorted(set(words))
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for i, word in enumerate(vocab)}

    def encode(self, text: str) -> list[int]:
        return [self.word_to_id[w] for w in text.split()]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_word[i] for i in ids]


# ─────────────────────────────────────────────
# 2. Self-Attention  (single head)
# ─────────────────────────────────────────────
class SelfAttention(nn.Module):
    """
    Masked (causal) self-attention.
    Kept as a standalone module — FFN lives in TransformerBlock.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.scale = embed_dim ** 0.5
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, embed_dim)
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        scores = Q @ K.transpose(-2, -1) / self.scale   # (seq_len, seq_len)

        # Causal mask — prevent attending to future tokens
        seq_len = x.shape[0]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        return weights @ V                               # (seq_len, embed_dim)


# ─────────────────────────────────────────────
# 3. Transformer Block
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """
    One Transformer block:
        Pre-LayerNorm → Self-Attention → residual
        Pre-LayerNorm → Feed-Forward   → residual

    Pre-norm (normalize *before* sub-layer) is more training-stable
    than the original post-norm design.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = SelfAttention(embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        # The 4× inner dimension follows the original Transformer paper
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer (pre-norm + residual)
        x = x + self.attn(self.norm1(x))
        # Feed-forward sub-layer (pre-norm + residual)
        x = x + self.ff(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# 4. Tiny LLM
# ─────────────────────────────────────────────
class TinyLLM(nn.Module):
    """
    vocab_size  – number of unique tokens
    embed_dim   – embedding / hidden dimension
    num_blocks  – number of stacked Transformer blocks
    max_seq_len – maximum sequence length for positional embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_blocks: int,
        max_seq_len: int = 100,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb   = nn.Embedding(max_seq_len, embed_dim)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim) for _ in range(num_blocks)]
        )

        # Final norm before the language-model head (common in modern LLMs)
        self.final_norm   = nn.LayerNorm(embed_dim)
        self.lm_head      = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len,)  integer token ids
        seq_len   = x.shape[0]
        positions = torch.arange(seq_len, device=x.device)

        h = self.token_emb(x) + self.pos_emb(positions)  # (seq_len, embed_dim)

        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)
        return self.lm_head(h)                            # (seq_len, vocab_size)


# ─────────────────────────────────────────────
# 5. Generation with temperature sampling
# ─────────────────────────────────────────────
def generate(
    model: TinyLLM,
    tokenizer: SimpleTokenizer,
    start_word: str,
    num_words: int = 5,
    temperature: float = 1.0,
) -> str:
    """
    Autoregressively generate `num_words` tokens after `start_word`.

    temperature < 1  → more focused / deterministic
    temperature = 1  → unmodified distribution
    temperature > 1  → more random / creative
    """
    model.eval()
    words = [start_word]

    for _ in range(num_words):
        ids = tokenizer.encode(" ".join(words))
        x   = torch.tensor(ids)

        with torch.no_grad():
            logits = model(x)           # (seq_len, vocab_size)

        # Take the last token's logits, apply temperature, then sample
        last_logits = logits[-1] / temperature
        probs       = F.softmax(last_logits, dim=-1)
        next_id     = torch.multinomial(probs, num_samples=1).item()

        words.append(tokenizer.id_to_word[next_id])

    return " ".join(words)


# ─────────────────────────────────────────────
# 6. Training script
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Tiny corpus ──────────────────────────
    text = "hello i am sujay you are"
    tokenizer = SimpleTokenizer(text)

    vocab_size = len(tokenizer.word_to_id)
    embed_dim  = 8
    num_blocks = 2

    print(f"Vocab size : {vocab_size}")
    print(f"Vocab      : {tokenizer.word_to_id}")

    # ── Model ────────────────────────────────
    # FIX: correct argument order → (vocab_size, embed_dim, num_blocks)
    model = TinyLLM(vocab_size, embed_dim, num_blocks)

    # Sanity-check forward pass
    all_ids = tokenizer.encode(text)
    x_test  = torch.tensor(all_ids)
    print(f"\nOutput shape: {model(x_test).shape}")  # (6, vocab_size)

    # ── Training ─────────────────────────────
    # Next-token prediction: input = ids[:-1], target = ids[1:]
    input_ids  = torch.tensor(all_ids[:-1])
    target_ids = torch.tensor(all_ids[1:])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("\nTraining…")
    for i in range(1000):
        logits = model(input_ids)                        # (seq_len-1, vocab_size)
        loss   = F.cross_entropy(logits, target_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"  Iter {i:4d} | loss = {loss.item():.4f}")

    # ── Generation ───────────────────────────
    print("\nGeneration (temperature=0.5, deterministic-ish):")
    print(generate(model, tokenizer, "hellox", num_words=5, temperature=0.5))

    print("\nGeneration (temperature=1.5, creative):")
    print(generate(model, tokenizer, "i", num_words=5, temperature=1.5))