# 자동 미분 흐름 예제
# 계산 흐름: a > b > c > out
# backward()를 통해  a < b < c < out을 계산하면 (out/a)'값이 a.grad에 채워짐
import torch

a = torch.ones(2, 2, requires_grad=True)

print(a)
print(a.data)
print(a.grad) # 계산한게 없으니 none
print(a.grad_fn) # 계산한게 없으니 none

b = a + 2
print(b)

c = b ** 2
print(c)

out =  c.sum()
print(out)

out.backward()

# 결과값 확인
print("------------a------------")
print(a.data) # 1,1,1,1
print(a.grad) # 6,6,6,6
print(a.grad_fn) # a에 대해 직접 계산한게 없으니 none


print("------------b------------")
print(b.data) 
print(b.grad) 
print(b.grad_fn) 


print("------------c------------")
print(c.data) 
print(c.grad) 
print(c.grad_fn) 


print("------------out------------")
print(out.data) 
print(out.grad) 
print(out.grad_fn) 