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


TRAIN_PATH = "C:/Users/user/Desktop/train"
VAL_PATH = "C:/Users/user/Desktop/val"
TEST_PATH = "C:/Users/user/Desktop/dicom_test"

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

model_dir = f"C:/Users/user/dicom/model/model_{version_num}"
os.makedirs(f"C:/Users/user/dicom/model/model_{version_num}", exist_ok=True) # mkdir 
config_path = f"C:/Users/user/dicom/model/model_{version_num}/config.json"
with open(config_path, 'w') as config_file:
    json.dump(config, config_file, indent=4)

print(f"Config saved to {config_path}")

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
])


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

log_dir = os.path.join(model_dir, "model_log")
weights_dir = os.path.join(model_dir, "weights_log")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

# 훈련 데이터 로딩
train_dataset = NiftiDataset(TRAIN_PATH, transform = transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = NiftiDataset(TEST_PATH, transform = transform)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = NiftiDataset(TEST_PATH, transform = transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

best_train_loss = float('inf')

with open(os.path.join(log_dir, "train_log.csv"), "w") as log:
    model.train()
    for epoch in range(num_epochs):

        current_train_loss = 0

        for step, image in enumerate(train_dataloader):
            image = image.type(torch.float32).to(device)
            patch_loss = 0

            for _ in range(num_patches):
                patch = generate_patch(image, patch_size)
                random_patch = deepcopy(patch) #추가
                inputs, coordinate = calculate_coordinate(random_patch, field_size, num_subpatches)       

                optimizer.zero_grad()
    
                outputs = model(inputs)
                outputs, targets = add_maskmap(outputs, coordinate)
                
                loss = criterion(outputs, targets)       
                loss.backward()
                optimizer.step()
                
                patch_loss += loss.item()
            
            patch_loss /= num_patches
        train_loss /= len(train_dataloader)

        # validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for step, image in enumerate(val_dataloader):
                image = image.type(torch.float32).to(device)
                patch_loss = 0

                for _ in range(num_patches):
                    patch = generate_patch(image, patch_size)
                    random_patch = deepcopy(patch)
                    inputs, coordinate = calculate_coordinate(random_patch, field_size, num_subpatches)
                    outputs = model(inputs)
                    outputs, targets = add_maskmap(outputs, coordinate)
                    loss = criterion(outputs, targets)
                    patch_loss += loss.item()
                
                val_loss += patch_loss / num_patches
            val_loss /= len(val_dataloader)
            
        print(f"Epoch: [{epoch}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        log.write(f"{epoch},{train_loss},{val_loss}\n")
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(weights_dir, f"best_model.pth"))

# Test Step
model.load_state_dict(torch.load(os.path.join(weights_dir, "best_model.pth")))
model.eval()
test_loss = 0
with torch.no_grad():
    for step, image in enumerate(test_dataloader):
        image = image.type(torch.float32).to(device)
        patch_loss = 0

        for _ in range(num_patches):
            patch = generate_patch(image, patch_size)
            random_patch = deepcopy(patch)
            inputs, coordinate = calculate_coordinate(random_patch, field_size, num_subpatches)
            outputs = model(inputs)
            outputs, targets = add_maskmap(outputs, coordinate)
            loss = criterion(outputs, targets)
            patch_loss += loss.item()
        
        test_loss += patch_loss / num_patches
    test_loss /= len(test_dataloader)
print(f"Final Test Loss: {test_loss:.4f}")

final_model_save_path = os.path.join(weights_dir, "final_model.pth")
torch.save(model.state_dict(), final_model_save_path)
print(f"Final model saved at {final_model_save_path}")


