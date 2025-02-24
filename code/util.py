#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# ## normalize

# In[19]:


def min_max_normalize(image):
    min_val = image.min()
    max_val = image.max()
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image


# In[20]:


def denormalize(image, min_val, max_val):
    denormalized_image = image * (max_val - min_val) + min_val
    return denormalized_image


# In[21]:


#예측을 위한 함수 정의
def denoise_image(model, image_path, slice_num):
    # 이미지 불러오기
    nifti = nib.load(image_path)
    image = nifti.get_fdata()

    image = image[:, :, slice_num]

    min_val = image.min()
    max_val = image.max()
    
    # normalize
    image = min_max_normalize(image)

    image_tensor = transforms.ToTensor()(image).unsqueeze(0).type(torch.float32)

    # 모델에 이미지 전달
    with torch.no_grad():
        model.eval()
        denoised_image = model(image_tensor)

    # denormalize
    denoised_image = denoised_image*max_val
    denoised_image = torch.clamp(denoised_image, min=0, max=max_val)

    # denormalize(denoised_image, min_val, max_val)
    
    return denoised_image.squeeze().cpu().numpy()


# ## patch

# In[22]:


def sampler(folder_path):
    """image slice"""
    file_list = os.listdir(folder_path)
    img_list = []
    for i in range(len(file_list)):
        file_path = os.path.join(folder_path, file_list[i])
        nifti = nib.load(file_path)
        image = nifti.get_fdata()
        
        for j in range(32):
            slice = image[:, :, j]
            img_list.append(slice)
    
    return img_list


# In[23]:


def generate_patch(image, patch_size): # 랜덤하게 patch 생성
    
    # 이미지 크기
    height, width = image.shape[2], image.shape[3]
    
    # 무작위한 패치의 시작점 결정
    top_left_x = np.random.randint(30, width - patch_size - 80 + 1)
    top_left_y = np.random.randint(0, height - patch_size + 1)

    # 이미지에서 무작위 패치 추출
    random_patch = image[:, :, top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size]

    return random_patch


# In[24]:


def generate_subpatch(patch, patch_size): # 랜덤하게 subpatch 생성
    
    # 이미지 크기
    height, width = patch.shape[2], patch.shape[3]
    
    # 무작위한 패치의 시작점 결정
    top_left_x = np.random.randint(0, width - patch_size + 1)
    top_left_y = np.random.randint(0, height - patch_size + 1)

    # 이미지에서 무작위 패치 추출
    random_subpatch = patch[:, :, top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size]

    return [random_subpatch, top_left_x, top_left_y] # patch 기준


# In[25]:


def calculate_coordinate(patch, field_size, num_subpatches): # 패치 내에서 랜덤하게 receptive field 생성
    # 64x64 크기의 패치에서 5x5 크기의 receptive field 추출
    coordinate = []
    replaced_value = []

    for i in range(num_subpatches):
        subpatch = generate_subpatch(patch, field_size)
        random_subpatch, top_row, top_col = subpatch[0], subpatch[1], subpatch[2]

        # center pixel 좌표
        center_row = top_row + (field_size // 2)
        center_col = top_col + (field_size // 2)
        center_pixel_value = patch[:, :, center_row, center_col]

        replaced_x, replaced_y = 2, 2
        # receptive field 내의 랜덤한 pixel 추출해서 대체
        while (replaced_x == 2):
            replaced_x = np.random.choice(5, replace=False)
        while (replaced_y == 2):
            replaced_y = np.random.choice(5, replace=False)

        replaced_value.append(patch[:, :, top_row + replaced_x, top_col + replaced_y])
        coordinate.append([center_row, center_col, center_pixel_value])
    
    for i in range(len(coordinate)):
        row, col, replaced = coordinate[i][0], coordinate[i][1], replaced_value[i]
        patch[:, :, row, col] = replaced
        
    return patch, coordinate


# In[26]:


def add_maskmap(outputs, coordinate):
    batch = outputs.shape[0]

    maskmap = torch.zeros([batch, 1, 64, 64], dtype=torch.float32)
    targets = torch.zeros([batch, 1, 64, 64], dtype=torch.float32)

    for i in range(len(coordinate)):
        row, col, value = coordinate[i][0], coordinate[i][1], coordinate[i][2]
        maskmap[:, :, row, col] = 1
        targets[:, :, row, col] = value
    
    
    maskmap = maskmap.to('cuda')
    targets = targets.to('cuda')
    
    outputs = outputs * maskmap

    return outputs, targets

