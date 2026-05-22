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
        seq_len = x.shape[0]
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        scores = Q@ K.transpose(-2,-1) / ((self.embed_dim)**0.5)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim =-1)
        output = weights @ V
        ff_output = self.ff(output)
        return self.norm(ff_output + output)
class TinyLLM(nn.Module):
    def __init__(self, embed_dim, num_blocks):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.pos_embedding = nn.Embedding(100, embed_dim)
        self.blocks = nn.ModuleList(
            [SelfAttention(embed_dim) for _ in range(num_blocks)]
        )
        self.final_linear = nn.Linear(embed_dim, vocab_size)
        

    def forward(self, x):
        positions = torch.arange(len(x))   # [0, 1, 2, 3, 4]
        x = self.embedding(x) + self.pos_embedding(positions)
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
print(output.shape)

ids = tokenizer.encode("the cat sat on the mat")
input_ids  = torch.tensor(ids[:-1])   # [4, 0, 3, 2, 5]
target_ids = torch.tensor(ids[1:])    # [0, 3, 2, 5, 1]

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for i in range(1000):
    output = model(input_ids)         # (5, vocab_size)
    loss = F.cross_entropy(output, target_ids)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f"Iter {i}: output = {output.detach()}loss={loss.item():.4f}")

def generate(model, tokenizer, start_word, num_words=5):
    model.eval()
    words = [start_word]
    
    for _ in range(num_words):
        ids = tokenizer.encode(" ".join(words))
        x = torch.tensor(ids)
        
        with torch.no_grad():
            output = model(x)
        
        # get last token's prediction
        last_scores = output[-1]
        next_id = torch.argmax(last_scores).item()
        next_word = tokenizer.id_to_word[next_id]
        words.append(next_word)
    
    return " ".join(words)

print(generate(model, tokenizer, "the"))