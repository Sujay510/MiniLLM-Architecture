import torch

x = torch.tensor([2.0, 3.0, 4.0])
actual = torch.tensor(1.0)

w = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)

for i in range(100):
    y = torch.dot(w, x)
    loss = (y - actual) ** 2

    loss.backward()
    with torch.no_grad():
        w -= 0.01 * w.grad
        w.grad.zero_()

    #if i % 1 == 0 and i<20:
     #   print(f"Iter {i}: y={y.item():.4f}, loss={loss.item():.4f}")
print(w)