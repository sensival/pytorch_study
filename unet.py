import os
import numpy as np

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torchvision.datasets import VOCSegmentation

## 하이퍼파라미터
lr = 1e-3
batch_size = 4
num_epoch = 100
data_dir = './dataset'
ckpt_dit = './checkpoint' # 트레이닝된 데이터 저장
log_dir = './log' # 텐서보드 로그
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Unet 구현
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 파란색 화살표 conv 3x3, batch-normalization, ReLU
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        ## Contracting path, encoder part
        # encoder part 1
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        # 빨간색 화살표(Maxpool)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # encoder part 2
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        # 빨간색 화살표(Maxpool)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # encoder part 3
        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        # 빨간색 화살표(Maxpool)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # encoder part 4
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        # 빨간색 화살표(Maxpool)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # encoder part 5
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)



        ## Expansive path, Decoder part
        # Decoder part 5
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        # 초록 화살표(Up Convolution)
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        # Decoder part 4
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512) # encoder part에서 전달된 512 채널 1개를 추가
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        # 초록 화살표(Up Convolution)
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        # Decoder part 3
        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)
        
        # 초록 화살표(Up Convolution)
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        # Decoder part 2
        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        # 초록 화살표(Up Convolution)
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        # Decoder part 1
        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        # class에 대한 output을 만들어주기 위해 1x1 conv
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x): # x는 input image, nn.Module의 __call__은 forward 함수를 저절로 실행 
        enc1_1 = self.enc1_1(x) # nn.Sequential객체는 __call__ Mx를 가지고 있어서 함수처럼 호출가능
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)
        # print(pool3.size())
        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x


# 데이터셋 로드 및 전처리 설정
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Pascal VOC Segmentation 데이터셋 불러오기
train_dataset = VOCSegmentation(root=data_dir, year='2012', image_set='train', download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 모델, 손실 함수, 옵티마이저 설정
model = UNet().to(device)
criterion = nn.CrossEntropyLoss()  # 세그멘테이션을 위한 교차 엔트로피 손실 함수
optimizer = optim.Adam(model.parameters(), lr=lr)

# TensorBoard SummaryWriter 설정
writer = SummaryWriter(log_dir=log_dir)

# 모델 학습 루프
for epoch in range(num_epochs):
    model.train()  # 모델을 학습 모드로 설정
    epoch_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target['segmentation'].to(device)

        # target을 LongTensor로 변환 (CrossEntropyLoss를 위해)
        target = target.long()

        # 옵티마이저 초기화
        optimizer.zero_grad()
        
        # 모델 순전파
        output = model(data)
        
        # 손실 계산
        loss = criterion(output, target)
        
        # 역전파 및 가중치 갱신
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # 에폭마다 평균 손실 출력
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # TensorBoard에 손실 기록
    writer.add_scalar('Loss/train', avg_loss, epoch + 1)
    
    # 에폭마다 모델 저장
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pth"))

# 학습 종료 후 TensorBoard writer 종료
writer.close()