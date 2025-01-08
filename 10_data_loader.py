import torch

# dataloader와 dataset을 통해 batch_size, train 여부, transform 등을 인자로 넣어 데이터를 어떻게 load 할 것인지 정해줄 수 있음 
from torch.utils.data import Dataset, DataLoader

# 토치비전은 파이토치에서 제공하는 데이터셋들이 모여있는 패키지
# transforms: 전처리할 때 사용하는 메소드 제공, transforms에서 제공하는 클래스 이외는 일반적으로 클래스를 따로 만들어 전처리 단계를 진행
import torchvision.transforms as transforms
from torchvision import datasets

# DataLoader의 인자로 들어갈 transform을 미리 정의할 수 있고 Compose 를 통해 리스트안에 순서대로 전처리 진행
# ToTensor()를 하는 이유는 torchvision이 PIL Image 형태로만 입력을 받기 때문에 데이터처리를 위해서 tensor 형으로 변환
mnist_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5,), std=(1.0))])
trainset = datasets.MNIST(root='./mnist/',
                          train=True, download=False,
                          transform=mnist_transform)
testset = datasets.MNIST(root='./mnist/',
                          train=False, download=False,
                          transform=mnist_transform)

# DataLoader 는 전체 데이터를 보관했다가 실제 모델학습을 할 때 batch_size 크기만큼 데이터를 가져옴
train_loader =  DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0) # num_workers=0은 싱글 프로세싱
test_loader =  DataLoader(testset, batch_size=8, shuffle=True, num_workers=0)

dataiter = iter(train_loader)
images, labels = next(dataiter)  # 여기 수정!
print(images.shape, labels.shape) # torch.Size([8, 1, 28, 28]) torch.Size([8])

torch_image = torch.squeeze(images[0])
print(torch_image.shape)
