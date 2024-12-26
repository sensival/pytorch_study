import torch

x = torch.ones(3,3, requires_grad=True)
print(x)

y = x + 5
print(y) # grad_fn=<AddBackward0>

z = y * y
out = z.mean()
print(z, out) #  grad_fn=<AddBackward0>),  grad_fn=<MeanBackward0>

# 계산이 완료된 후 .backward()를 호출하면 자동으로 역전파 계산이 가능하고 , grad 속성에 누적됨

print(out)
out.backward()

# grad: data가 거쳐온 layer에 대한 미분값 저장

print(x)
print(x.grad)

x = torch.randn(3, requires_grad=True)
print(x)
y = x * 2
while y.data.norm() < 1000: # norm은 텐서의 유클리드 노름
    print(y.data.norm())
    y = y * 2

print(y)