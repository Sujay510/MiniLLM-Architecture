import torch

x = torch.tensor([2.0, 3.0, 4.0])
actual = torch.tensor([1.0,2.0])

w = torch.tensor([[0.5, 0.5, 0.5],
                 [0.25, 0.25, 0.25]], requires_grad=True)
b = torch.tensor([0.0, 0.0], requires_grad = True)
for i in range(100):
    y = w@x + b
    loss = ((y - actual) ** 2).sum()

    loss.backward()
    with torch.no_grad():
        w -= 0.01 * w.grad
        w.grad.zero_()

    if i % 1 == 0 and i<11:
       print(f"Iter {i}: y={y.detach()}, loss={loss.item():.4f}")
