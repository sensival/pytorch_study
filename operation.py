import math
import torch


a = torch.rand(1, 2) * 2 - 1
print(a)
print(torch.abs(a))
print(torch.ceil(a))
print(torch.floor(a))
print(torch.clamp(a, -0.5, 0.5)) # 최대 최소값 제한
print(torch.min(a))
print(torch.max(a))
print(torch.mean(a))
print(torch.std(a))
print(torch.prod(a))
print(torch.unique(torch.tensor([1, 2, 3, 1, 2, 2])))

# 최대 최소
x = torch.rand(2, 2)
print(x)
print(x.max(dim=0)) # 열이 같은 것 중 선택
print(x.max(dim=1)) # 행이 같은 것 중 선택



x = torch.rand(2, 2)
print(x)
y = torch.rand(2, 2)
print(y)

# add, sub, mul, div
print(torch.add(x, y))
result = torch.empty(2,4)
torch.add(x, y, out=result)
print(result)

# 더하고 덮어쓰기 in-place
y.add_(x)

# 내적
print(x)
print(y)
print(torch.matmul(x, y))
z = torch.mm(x, y)
print(z)
print(torch.svd(z))
# U: 입력 행렬과 같은 크기의 직교 행렬 Q ^TQ=QQ^T=I 인 Q
# S: 단일값을 포함한 1차원 텐서(대각행렬의 가운데 값들 나열)
# V: A의 전치된 직교 행렬