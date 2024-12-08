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
# max와 min은 dim 인자를 줄 경우 argmax, argmin도 리턴 argmax(최대값을 가진 인덱스), argmin(최소값을 가진 인덱스)



x = torch.rand(2, 2)
print(x)
y = torch.rand(2, 2)
print(y)

# add, sub, mul, div
print(torch.add(x, y))
result = torch.empty(2,4)
torch.add(x, y, out=result)
print(result)

# 더하고 덮어쓰기 in-place _ 언더바 붙이기
y.add_(x) #y에 더한 값이 덮어 씌워짐
y.sub_(x)
y.mul_(x)
y.div_(x)

# 내적(dot product)
print(x)
print(y)
print(torch.matmul(x, y))
z = torch.mm(x, y)
print(z)
print(torch.svd(z)) #행렬분해
# U: 입력 행렬과 같은 크기의 직교 행렬 Q ^TQ=QQ^T=I 인 Q
# S: 단일값을 포함한 1차원 텐서(대각행렬의 가운데 값들 나열)
# V: A의 전치된 직교 행렬


# 인덱싱
x = torch.tensor([[1,2], [3,4]])
print(x[0,0])
print(x[0,1])
print(x[1,0])
print(x[1,1])

# 슬라이싱
print(x[:,0]) # 1, 3
print(x[:,1]) # 2, 4

# 크기나 모양변경
x= torch.randn(4,5)
print(x)
y=x.view(20)
print(y)
z=x.view(5,-1) # -1은 나머지는 너가 알아서 정해줘~ -> 5x4
print(z)

# 스칼라 값 출력
x = torch.randn(1) # 2면 item 오류
print(x)
print(x.item()) 
print(x.dtype)



