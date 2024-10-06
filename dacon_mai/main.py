import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2
from ultralytics import YOLO 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transform_unet import UNet

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

'''
class BaseModel(nn.Module):
    def __init__(self, gene_size=CFG['label_size'], model_path='yolo11n-cls.pt'):
        super(BaseModel, self).__init__()
        self.backbone = YOLO(model_path).model  # YOLO11n 분류 모델 로드
        self.backbone.model[-1] = nn.Identity()  # 최종 분류 레이어 제거
        
        # 회귀 레이어 추가 (유전체 발현 예측용)
        self.regressor = nn.Linear(1000, gene_size)  # 1280은 특징 추출기의 출력 크기

    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        return x

'''

def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.MSELoss().to(device)
    
    best_loss = 99999999
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
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
                    
        _val_loss = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}]')
       
        if scheduler is not None:
            scheduler.step(_val_loss)
            
        if best_loss > _val_loss:
            best_loss = _val_loss
            best_model = model
    
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
        
        _val_loss = np.mean(val_loss)
    
    return _val_loss



model = UNet(gene_size=CFG['label_size'])
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)


test = pd.read_csv('./test.csv')
test_dataset = CustomDataset(test['path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(test_loader):
            imgs = imgs.to(device).float()
            pred = model(imgs)
            
            preds.append(pred.detach().cpu())
    
    preds = torch.cat(preds).numpy()

    return preds


preds = inference(infer_model, test_loader, device)



submit = pd.read_csv('./sample_submission.csv')
submit.iloc[:, 1:] = np.array(preds).astype(np.float32)
submit.to_csv('./baseline_submit.csv', index=False)