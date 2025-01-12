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