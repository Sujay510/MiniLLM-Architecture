import torch
import torch.nn as nn
import torch.nn.functional as F
import bytepairencoder as bpe
import os

def batch(ids,size,batch_size):
    input_list = []
    target_list = []
    for _ in range(batch_size):
        start = torch.randint(0, len(ids) - size, (1,)).item()
        chunk = ids[start:start+size+1]
        input_ids1 = torch.tensor(chunk[:-1])
        target_ids1 = torch.tensor(chunk[1:])
        input_list.append(input_ids1)
        target_list.append(target_ids1)
    input_ids = torch.stack(input_list) 
    target_ids = torch.stack(target_list) 
    return input_ids, target_ids

def top_k_sample(logits: torch.Tensor, k: int, temperature: float = 1.0) -> int:
    if temperature <= 0: raise ValueError
    if logits.dim() != 1: raise ValueError
    if k < 0: raise ValueError
    x, indices = torch.sort(logits, descending=True)
    k = min(logits.size(0), k)
    y = x[:k] / temperature
    result = torch.softmax(y, dim=0)
    sample = torch.multinomial(result, num_samples=1)
    return indices[sample].item()

class Simple_Tokenizer:
    def __init__(self,tokens,merges):
        self.merges = merges
        vocab = sorted(list(set(tokens)))
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for i, word in enumerate(vocab)}

    def encode(self,text):
        tokens = bpe.encode(text,self.merges)
        return [self.word_to_id[token] for token in tokens]
    def decode(self,ids):
        tokens = [self.id_to_word[id] for id in ids]
        return bpe.decode(tokens)
    
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
        seq_len = x.shape[1]
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        scores = Q@ K.transpose(-2,-1) / ((self.embed_dim)**0.5)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim =-1)
        output = weights @ V
        x = self.norm(output + x)
        ff_output = self.ff(x)
        return self.norm(ff_output + x)
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim,num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim,num_heads)
        self.ff = nn.Sequential(
                    nn.Linear(embed_dim, 4 * embed_dim),
                    nn.ReLU(),
                    nn.Linear(4 * embed_dim, embed_dim)
                    )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        output = self.attention(x)
        x = self.norm1(output + x)
        ff_output = self.ff(x)
        return self.norm2(ff_output + x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.Wo = nn.Linear(embed_dim, embed_dim)
    
    def forward(self,x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = Q.view(batch_size,seq_len,self.num_heads,self.head_dim)
        K = K.view(batch_size,seq_len,self.num_heads,self.head_dim)
        V = V.view(batch_size,seq_len,self.num_heads,self.head_dim)
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        scores = Q@K.transpose(-2,-1)/((self.head_dim)**0.5)
        mask = torch.triu(torch.ones(seq_len,seq_len),diagonal=1).bool()
        scores = scores.masked_fill(mask,float('-inf'))
        weights = F.softmax(scores,dim=-1)
        output = weights@V
        output = output.reshape(batch_size,seq_len,self.embed_dim)
        output = self.Wo(output)
        return output
class TinyLLM(nn.Module):
    def __init__(self,vocab_size, embed_dim, num_blocks,num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.pos_embedding = nn.Embedding(100, embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim,num_heads) for _ in range(num_blocks)]
        )
        self.final_linear = nn.Linear(embed_dim, vocab_size)
        

    def forward(self, x):
        positions = torch.arange(x.shape[1])  
        x = self.embedding(x) + self.pos_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.final_linear(x)

with open("shakespeare.txt") as f:  
    text = f.read()
merges = bpe.train_bpe(text,num_merges=700)
tokens = bpe.encode(text,merges)
tokenizer = Simple_Tokenizer(tokens,merges)

# encode input
ids = [tokenizer.word_to_id[t] for t in tokens]
split_pt = int(0.9*len(ids))
train_ids = ids[:split_pt]
val_ids = ids[split_pt:]
# pass through model
vocab_size = len(tokenizer.word_to_id)
embed_dim = 64
num_blocks = 4

model = TinyLLM(vocab_size,embed_dim,num_blocks,num_heads=8)
input_ids,target_ids = batch(train_ids,size=32,batch_size=16)
output = model(input_ids)
print(output.shape)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

if os.path.exists("model.pt"):
    model.load_state_dict(torch.load("model.pt"))
    optimizer.load_state_dict(torch.load("optimizer.pt"))
    print("Checkpoint Loaded")
else:
    print("Fresh Start")

for i in range(1000):
    input_ids,target_ids = batch(train_ids,size=32,batch_size=16)
    output = model(input_ids)
    output = output.reshape(-1,vocab_size)
    target_ids = target_ids.reshape(-1)
    loss = F.cross_entropy(output, target_ids)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        val_input,val_target = batch(val_ids,size=32,batch_size=16)
        with torch.no_grad():
            val_output = model(val_input)
            val_output = val_output.reshape(-1,vocab_size)
            val_target = val_target.reshape(-1)
            val_loss = F.cross_entropy(val_output,val_target)
        print(f"Iter {i}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")

torch.save(model.state_dict(),"model.pt")
torch.save(optimizer.state_dict(),"optimizer.pt")

def generate(model, tokenizer, start_word, num_words=50,block_size=32):
    model.eval()
    ids = tokenizer.encode(start_word)
    
    for _ in range(num_words):
        context = ids[-block_size:]
        x = torch.tensor(context).unsqueeze(0)
        
        with torch.no_grad():
            output = model(x)
        
        # get last token's prediction
        last_scores = output[0,-1]
        next_id = top_k_sample(last_scores, k=3, temperature=0.7)
        print(next_id, tokenizer.id_to_word[next_id])
        ids.append(next_id)
    
    return tokenizer.decode(ids)
seed_text = "Hello, Who are you?"
print(generate(model, tokenizer, seed_text,num_words=50,block_size=32))

