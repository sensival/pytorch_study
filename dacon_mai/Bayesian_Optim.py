import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader



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
    'EPOCHS':1,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':32,
    'SEED':42
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

def train_model(model, optimizer, train_loader, device):
    model.train()
    criterion = torch.nn.MSELoss()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_model(model, val_loader, device):
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            loss = criterion(output, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def objective(trial):
    # 튜닝할 하이퍼파라미터 정의
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # 데이터로더 정의
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 모델, 옵티마이저 정의
    model = BaseModel().to(device)  # input_dim과 output_dim은 예시에 맞게 설정하세요.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 학습
    for epoch in range(10):  # 예시로 10 epoch 동안 학습
        train_loss = train_model(model, optimizer, train_loader, device)
    
    # 검증 손실 계산
    val_loss = evaluate_model(model, val_loader, device)
    return val_loss




# Bayesian Optimization 실행
study = optuna.create_study(direction='minimize')  # 손실을 최소화하는 방향으로 탐색
study.optimize(objective, n_trials=50)  # 50번 시도

# 최적의 하이퍼파라미터 출력
print("Best hyperparameters:", study.best_params)
print("Best validation loss:", study.best_value)
