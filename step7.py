import torch
import torch.nn as nn

# 10 possible words, each as a 4d vector
embedding = nn.Embedding(10, 4)

# look up word 3
token = torch.tensor(3)
vector = embedding(token)
print(vector)

# look up multiple words at once
tokens = torch.tensor([1, 3, 7])
vectors = embedding(tokens)
print(vectors.shape)
print(vectors)  # what do you expect this to be?