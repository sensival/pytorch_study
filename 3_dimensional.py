import torch

# 1D tensor(vector)
t1 = torch.tensor([1,2,3])
print(t1.ndim)
print(t1.shape)
print(t1)

# 2D tensor(matrix): 일반적인 통계 데이터셋
t2 = torch.tensor([[1,2,3],
                   [4,5,6],
                   [7,8,9]])
print(t2.ndim)
print(t2.shape)
print(t2)

# 3D tensor(cube): 시간 축이 포함된 시계열 데이터셋
t3 = torch.tensor([[[1,2,3],
                   [4,5,6],
                   [7,8,9]],
                   [[1,2,3],
                   [4,5,6],
                   [7,8,9]]])
print(t3.ndim)
print(t3.shape)
print(t3)

# 4D tensor: 컬러이미지(샘플, 높이, 너비, 컬러 채널) -> 흑백은 컬러채널 제외 3D 가능
# 5D tensor: 비디오(샘플, 높이, 너비, 컬러 채널, 프레임)

# 크기 모양 변경
x = torch.randn(4,5)
print(x)
y = x.view(20)
print(y)
z = x.view(5, -1) # 행은 5개 나머지는 알아서
print(z)

# squeeze 차원 축소: 차원이 1인 차원을 제거해준다. 
tensor = torch.rand(1, 3, 3)
print(tensor)
print(tensor.shape)
t = tensor.squeeze()
print(t)
print(t.shape) # 3, 3

# unsqueeze 차원 증가: 1인 차원을 생성하는 함수이다. 그래서 어느 차원에 1인 차원을 생성할 지 꼭 지정해주어야한다.
tensor = torch.rand(3, 3)
print(tensor)
print(tensor.shape)
t = tensor.unsqueeze(dim=0)
print(t)
print(t.shape) # 1 3 3

tensor = torch.rand(3, 3)
print(tensor)
print(tensor.shape)
t = tensor.unsqueeze(dim=2)
print(t)
print(t.shape) # 3 3 1