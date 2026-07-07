import torch

x=torch.tensor(2.0)
actual=torch.tensor(4.0)

w = torch.tensor(1.5, requires_grad=True)

for i in range(100):
    y = w * x
    loss = (y - actual) ** 2

    loss.backward()
    with torch.no_grad():
        w -= 0.01 * w.grad
        w.grad.zero_()
    if i % 10 == 0:
        print(f"Iteration {i}: w={w.item():.4f}, loss={loss.item():.4f}")
print(torch.cuda.is_available())
print(torch.backends.mps.is_available())