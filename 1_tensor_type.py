import torch

# 버전 확인
print(torch.__version__)

# 초기화 되지 않은 텐서
x = torch.empty(4, 2)
print(x)

# rand
x = torch.rand(4, 2)
print(x)

# long 형으로
x = torch.zeros(4, 2, dtype=torch.long)
print(x)

# 값지정 
x = torch.tensor([3, 2.3])
print(x)

# 2x4 1로 채워진 double
x = x.new_ones(2, 4, dtype=torch.double)
print(x)

# x와 같은 크기 float
x = torch.randn_like(x, dtype=torch.float)
print(x)

print(x.size())
print(torch.Size([2,4]))

# float tensor
ft = torch.FloatTensor([1,2,3])
print(ft)
print(ft.short())
print(ft.int()) # 캐스팅 가능 int -> float 형으로 가능