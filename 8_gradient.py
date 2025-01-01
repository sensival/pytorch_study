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

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

# with torch.no_grad()를 사용하여 기울기의 업데이트를 하지 않음
# 기록을 추적하는 것을 방지하기 위해 코드 블럭을 with torch.no_grad()로 감싸면 기울지 계산은 필요없지만,
# requires_grad=True로 설정되어 학습가능한 매개변수를 갖는 모델을 평가할 때 유용--> 평가할 때는 기울기 값을 업데이트하지 않으니까

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)


# detach():내용물은 같지만 requires_grad가 다른 새로운 tensor를 가져올 때

print(x.requires_grad) # true로 설정된 상태
y = x.detach() # requires_grad가 false인 상태로 copy
print(y.requires_grad) 
print(x.eq(y).all()) #  true