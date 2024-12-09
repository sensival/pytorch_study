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
t1, t2, t3 = torch.chunk(tensor, 3, dim = 1)
print(t1)
print(t2)
print(t2)
