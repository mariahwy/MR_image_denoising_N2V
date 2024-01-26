#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nibabel as nib
import json

import torch
import torch.nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from copy import deepcopy

# from layer import *
from unet import *
from util import *


# In[16]:


# 훈련 hyperparameter
TRAIN_PATH = "C:/Users/user/Desktop/인턴/train"

version_num = 7
num_epochs = 1000
batch_size = 256
patch_size = 64
field_size = 5
num_patches = 50
num_subpatches = 400
lr = 4e-3

# config 저장
config = {
    "model_num": version_num,
    "batch_size": batch_size,
    "learning_rate": lr,
    "num_epochs": num_epochs,
    "num_patches": num_patches,
    "num_subpatches": num_subpatches
}

os.makedirs(f"C:/Users/user/dicom/model/model_{version_num}", exist_ok=True) # mkdir 
config_path = f"C:/Users/user/dicom/model/model_{version_num}/config.json"
with open(config_path, 'w') as config_file:
    json.dump(config, config_file, indent=4)

print(f"Config saved to {config_path}")

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
])


# In[17]:


TEST_PATH = "C:/Users/user/dicom/data/dicom_test"


# In[18]:


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# ## train

# In[28]:


# 모델 인스턴스 생성
model = UNet_model(1, 1, False)
model.to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

log_dir = f"C:/Users/user/dicom/model/model_{version_num}/model_log" # window에서는 역슬래시
weights_dir = f"C:/Users/user/dicom/model/model_{version_num}/weights_log"
os.makedirs(log_dir, exist_ok=True) # mkdir
os.makedirs(weights_dir, exist_ok=True) # mkdir

# 훈련 데이터 로딩
train_dataset = NiftiDataset(TEST_PATH, transform = transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

best_train_loss = float('inf')

with open(os.path.join(log_dir, "train_log.csv"), "w") as log:
    model.train()
    for epoch in range(num_epochs):

        current_train_loss = 0

        for step, image in enumerate(train_dataloader):
            image = image.type(torch.float32)
            image = image.to(device)
            
            patch_loss = 0

            for i in range(num_patches):
                patch = generate_patch(image, patch_size)
                random_patch = deepcopy(patch) #추가
                inputs, coordinate = calculate_coordinate(random_patch, field_size, num_subpatches)       

                optimizer.zero_grad()

                # 모델에 입력 전달 및 출력 얻기        
                outputs = model(inputs)
                outputs, targets = add_maskmap(outputs, coordinate)
                # print(outputs.shape, targets.shape)
                # print(torch.min(outputs), torch.max(outputs))
                # print(torch.min(targets), torch.max(targets))
                # print('='*50)
                # 손실 계산 및 역전파
                loss = criterion(outputs, targets)       
                loss.backward()
                optimizer.step()
                
                patch_loss += loss.item()
            
            patch_loss /= num_patches

            current_train_loss += patch_loss
        
        current_train_loss /= len(train_dataloader)

        print(f"Iter: [{epoch}/{num_epochs}] | Train Loss: {current_train_loss}\n")
            
        if current_train_loss < best_train_loss:
            best_train_loss = current_train_loss
            torch.save(model.state_dict(), f'C:/Users/user/dicom/model/model_{version_num}/weights_log/model_{epoch:04d}.pth')
            log.write(f"{epoch},{current_train_loss}\n")
        else:
            pass

final_model_save_path = f"C:/Users/user/dicom/model/model_{version_num}/weights_log/final_model.pth"
torch.save(model.state_dict(), final_model_save_path)
print(f"Final model saved at {final_model_save_path}")

