import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple_Tokenizer:
    def __init__(self,text):
        words = text.split()
        vocab = sorted(list(set(words)))
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for i, word in enumerate(vocab)}

    def encode(self,text):
        words = text.split()
        return [self.word_to_id[word] for word in words]
    def decode(self,ids):
        return [self.id_to_word[id] for id in ids]
    
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
    
text = "the cat sat on the mat"
tokenizer = Simple_Tokenizer(text)

# encode input
ids = tokenizer.encode("the cat sat")
x = torch.tensor(ids)

# pass through model
vocab_size = len(tokenizer.word_to_id)
embed_dim = 8
num_blocks = 2

model = TinyLLM(embed_dim, num_blocks)
output = model(x)
print(output.shape)   # (3, vocab_size)