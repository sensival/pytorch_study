# 신경망 구성
# 레이어(layer): 신경망의 핵심 데이터 구조로 하나 이상의 텐서를 입력 받아 하나이상의 텐서를 출력
# 모듈(module): 한개 이상의 계층이 모여서 구성
# 모델: 한개 이상의 모듈이 모여서 구성

import torch

# torch.nn 패키지
# 주로 가중치(weight), 편향(bias)값들이 내부에서 자동으로 생성되는 레이어들을 사용할 때 사용(weight 값을 직접선언 안함)

import torch.nn as nn

# nn.Linear 계층 예제

input = torch.randn(128, 20)
print(input)

n = nn.Linear(20, 30)
print(n)

output = n(input)
print(output)
print(output.size())

# 입력 (128, 20) → nn.Linear(20, 30) 계층 → 출력 (128, 30)
# 선형 변환을 수행하며 내부적으로 W * input^T + b 연산이 이루어짐.
# nn.Linear 계층은 학습 가능한 가중치와 편향을 포함함.

# nn.Conv2d

input = torch.randn(20, 16, 50, 100)
print(input.size())
m = nn.Conv2d(16, 33, 3, stride=2)
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
print(m)

# dilation은 커널 내부의 요소 간격을 조절하는 옵션입니다. 기본적으로 dilation=1이면 커널의 요소들이 연속된 픽셀을 대상으로 계산합니다. 하지만 dilation > 1이면 커널 내부에서 요소들이 일정한 간격을 띄고 샘플링하게 됩니다.
# 설정	커널이 커버하는 실제 크기
# dilation=(1,1) (기본값)	(3,5)
# dilation=(2,1)	(5,5)
# dilation=(3,1)	(7,5)

output = m(input)
print(output.size())
# torch.Size([20, 16, 50, 100])
# Conv2d(16, 33, kernel_size=(3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
# torch.Size([20, 33, 26, 100])