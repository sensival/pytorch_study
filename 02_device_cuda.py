import torch

x = torch.rand(1)
print(x)
print(x.item())
print(x.dtype)

torch.device('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
y = torch.ones_like(x, device=device)
print(y)
x = x.to(device)
print(x)
z = x + y
print(z)
print(z.to('cpu', torch.double))