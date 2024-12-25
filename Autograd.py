# 자동미분
# tensor의 모든 연산에 대해 자동미분 제공
# 코드를 어땋게 작성하느냐에 따라 역전파가 정의됨
# back prop을 위해 미분값 자동 계산

import torch

#------------------------------------------
# requires_grad 속성을 True로 설정하면, 해당 텐서에서 이루어지는 모든 연산들을 추적하기 시작
# 기록을 추적하는 것을 중단하게 하려면, detach()를 호출하여 연산기록으로부터 분리

a = torch.randn(3,3)
a = a * 3
print(a)
print(a.requires_grad) # false 가 기본설정


# requires_grad_(...)는 기존 텐서의 requies_grad 값을 바꿔치기(in-place)하여 변경
# grad_fn: 미분갑을 계산한 함수에 대한 정보 저장(어떤 함수에 대해서 backprop 했는지)


a.requires_grad_(True) # 언더바는 in-place 연산
print(a.requires_grad)

b = (a * a).sum() 
print(b)
print(b.grad_fn) # =<SumBackward0> ->sum() 연산을 했다는 것을 기억