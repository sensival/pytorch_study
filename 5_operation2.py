import math
import torch

# 텐서 간 결합 stack
x = torch.FloatTensor([1,4])
print(x)
y = torch.FloatTensor([2,5])
print(y)
z = torch.FloatTensor([3,6])
print(z)
print(torch.stack([x,y,z]))

# 텐서 간 결합 cat ---> dim이 존재해야함
a = torch.randn(1,3,3)
print(a)
b = torch.randn(1,3,3)
print(b)
c= torch.cat((a,b), dim = 1)
print(c)
print(c.size()) # dim = 0 -> 2,3,3, dim = 1 -> 1,6,3


# 텐서 나누기 chunk 53:58
tensor = torch.rand(3,6)
print(tensor)
t1, t2, t3 = torch.chunk(tensor, 3, dim = 1) # 3개로 나눌거야
print(t2)
print(t2)

# spilt: chunk와 동일하지만 조금 다름
tensor =torch.rand(3,6)
t1, t2 = torch.split(tensor, 3, dim=1) # 열이 3개로 나눠줘
print(t1)
print(tensor)
print(t1)
print(t2)

