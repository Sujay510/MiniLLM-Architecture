import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        scores = Q@ K.T
        weights = F.softmax(scores, dim =-1)
        output = weights @ V
        return self.norm(output + x)
    
x = torch.randn(3, 4)   # 3 tokens, 4d embeddings
attn = SelfAttention(embed_dim=4)
output = attn(x)
print(output.shape)      # should be (3, 4)