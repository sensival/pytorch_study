import math
import torch
import numpy as np

a =  torch.ones(7)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b) # b도 2,2,2..로 바뀜 d/t 메모리 공유


# 토치와 넘파이는 메모리 공유
a =  torch.ones(7)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a) # 이거도 둘다 2,2,2,2,...