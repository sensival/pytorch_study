
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore') 


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


CFG = {
    'IMG_SIZE':224,
    'EPOCHS':5,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':32,
    'SEED':41
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정



df = pd.read_csv('./train.csv')
train_len = int(len(df) * 0.8)
train_df = df.iloc[:train_len]
val_df = df.iloc[train_len:]

train_label_vec = train_df.iloc[:,2:].values.astype(np.float32)
val_label_vec = val_df.iloc[:,2:].values.astype(np.float32)

CFG['label_size'] = train_label_vec.shape[1]


class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        image = cv2.imread(img_path)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)
    




train_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

test_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])


train_dataset = CustomDataset(train_df['path'].values, train_label_vec, train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

val_dataset = CustomDataset(val_df['path'].values, val_label_vec, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)



class BaseModel(nn.Module):
    def __init__(self, gene_size=CFG['label_size']):
        super(BaseModel, self).__init__()
        self.backbone = models.resnet152(pretrained=True)
        self.regressor = nn.Linear(1000, gene_size)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        return x


def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.MSELoss().to(device)
    
    best_loss = float('inf')
    best_model = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []
        
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
        
        _train_loss = np.mean(train_loss)
        train_losses.append(_train_loss)
        
        _val_loss = validation(model, criterion, val_loader, device)
        val_losses.append(_val_loss)
        
        print(f'Epoch [{epoch}], Train Loss: [{_train_loss:.5f}], Val Loss: [{_val_loss:.5f}]')
        
        if scheduler is not None:
            scheduler.step(_val_loss)
        
        # best model 저장
        if best_loss > _val_loss:
            best_loss = _val_loss
            best_model = model
            torch.save(best_model.state_dict(), './best_model.pth')
    
    # 학습 손실 그래프 그리기 및 저장
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train and Validation Loss Over Epochs")
    plt.savefig("./loss_curve.png")
    plt.show()
    
    return best_model

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            pred = model(imgs)
            loss = criterion(pred, labels)
            val_loss.append(loss.item())
    
    return np.mean(val_loss)


# 1. 모델 불러오기
model = BaseModel()  # 모델 구조를 정의했던 클래스 또는 함수로 모델을 초기화합니다.
model.load_state_dict(torch.load('best_model3.pth', map_location=device))  # 모델 가중치를 로드합니다.
model.to(device)
model.eval()  # 평가 모드로 설정합니다.

# 2. 테스트 데이터 로딩
test = pd.read_csv('./test.csv')
test_dataset = CustomDataset(test['path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

# 3. 예측 함수 정의
def inference(model, test_loader, device):
    preds = []
    with torch.no_grad():
        for imgs in tqdm(test_loader):
            imgs = imgs.to(device).float()
            pred = model(imgs)
            preds.append(pred.detach().cpu())
    
    preds = torch.cat(preds).numpy()
    return preds

# 4. 예측 수행
preds = inference(model, test_loader, device)

# 5. 제출 파일 생성
submit = pd.read_csv('./sample_submission.csv')
submit.iloc[:, 1:] = np.array(preds).astype(np.float32)
submit.to_csv('./baseline_submit3.csv', index=False)
