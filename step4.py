import torch

x = torch.tensor([2.0, 3.0, 4.0])
actual = torch.tensor([1.0])

w1 = torch.tensor([[0.5, 0.5, 0.5],
                 [0.25, 0.25, 0.25]], requires_grad=True)
w2 = torch.tensor([0.5,0.5], requires_grad=True )
b1 = torch.tensor([0.0, 0.0], requires_grad = True)
b2 = torch.tensor([0.0], requires_grad=True)
for i in range(100):
    y1 = w1@x + b1
    y2 = w2@y1 + b2
    loss = ((y2 - actual) ** 2).sum()

    loss.backward()
    with torch.no_grad():
        w1 -= 0.01 * w1.grad
        b1 -= 0.01 * b1.grad
        w2 -= 0.01 * w2.grad
        b2 -= 0.01 * b2.grad
        w1.grad.zero_()
        b1.grad.zero_()
        w2.grad.zero_()
        b2.grad.zero_()
    if i % 10 == 0:
       print(f"Iter {i}: y1={y1.detach()}, y2={y2.item()}, loss={loss.item():.4f}")
