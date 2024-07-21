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

