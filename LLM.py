import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.ff = nn.Sequential(
                    nn.Linear(embed_dim, 4 * embed_dim),
                    nn.ReLU(),
                    nn.Linear(4 * embed_dim, embed_dim)
                    )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        scores = Q@ K.T / ((self.embed_dim)**0.5)
        weights = F.softmax(scores, dim =-1)
        output = weights @ V
        ff_output = self.ff(output)
        return self.norm(ff_output + output)
class TinyLLM(nn.Module):
    def __init__(self, embed_dim, num_blocks):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.blocks = nn.ModuleList(
            [SelfAttention(embed_dim) for _ in range(num_blocks)]
        )
        self.final_linear = nn.Linear(embed_dim, vocab_size)
        

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.final_linear(x)
    
vocab_size = 1000
embed_dim = 4
num_blocks = 3

model = TinyLLM(embed_dim, num_blocks)

x = torch.tensor([1, 3, 7])   # 3 token ids
output = model(x)
print(output.shape)            # should be (3, 1000)