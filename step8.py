import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

x = torch.randn(3, 4)  # 3 tokens, 4d embeddings

Wq = torch.randn(4, 4)
Wk = torch.randn(4, 4)
Wv = torch.randn(4, 4)

Q = x@Wq
K = x@Wk
V = x@Wv

scores = Q@ K.T
weights = F.softmax(scores, dim =-1)
output = weights @ V

print(output.shape)
print(weights)