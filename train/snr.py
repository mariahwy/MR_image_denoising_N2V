#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

from unet import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[2]:


def calculate_snr(image_num, signal_roi, noise_roi):
    
    signal_sum = 0
    noise_sum = 0
    sum = 0

    for i in range(32):
        image = image_num[:, :, i]

        for j in range(3):
            signal = image[signal_roi[j][0]:signal_roi[j][0]+20, signal_roi[j][1]:signal_roi[j][1]+20]
            noise = image[noise_roi[j][0]:noise_roi[j][0]+20, noise_roi[j][1]:noise_roi[j][1]+20]

            signal = np.mean(signal)
            noise = np.std(noise)

            signal_sum += signal
            noise_sum += noise
       
        mean = signal_sum / 3
        std = noise_sum / 3

        # SNR 계산
        snr = mean / std

        sum += snr

    avg_snr = sum / 32

    return avg_snr


# In[7]:


PATH = "C:/Users/user/dicom/model/model_5/output"

# model.load_state_dict(torch.load("C:/Users/user/dicom/model/model_4/weights_log/model_0071.pth")) 

image_list = os.listdir(PATH)


# In[8]:


noise_roi = [[10, 10], [10, 325], [350, 340]]
signal_roi = [[260, 210], [125, 195], [160, 280]]


# In[104]:


IMAGE_NUM = image_list[13]  # 0~34

image_path = os.path.join(PATH, IMAGE_NUM)
nifti = nib.load(image_path)
image = nifti.get_fdata()


# In[105]:


plt.figure(figsize=(5, 5))

plt.subplot(1, 1, 1)
plt.imshow(image[:, :, 16].T, cmap='gray', origin = 'lower')
plt.title('ROI (red = noise, blue = signal)')

shp1=patches.Rectangle((signal_roi[0][0], signal_roi[0][1]), 20, 20, color='b', fill = False)
plt.gca().add_patch(shp1)
shp2=patches.Rectangle((signal_roi[1][0], signal_roi[1][1]), 20, 20, color='b', fill = False)
plt.gca().add_patch(shp2)
shp3=patches.Rectangle((signal_roi[2][0], signal_roi[2][1]), 20, 20, color='b', fill = False)
plt.gca().add_patch(shp3)

shp4=patches.Rectangle((noise_roi[0][0], noise_roi[0][1]), 20, 20, color='r', fill = False)
plt.gca().add_patch(shp4)
shp5=patches.Rectangle((noise_roi[1][0], noise_roi[1][1]), 20, 20, color='r', fill = False)
plt.gca().add_patch(shp5)
shp6=patches.Rectangle((noise_roi[2][0], noise_roi[2][1]), 20, 20, color='r', fill = False)
plt.gca().add_patch(shp6)


# In[68]:


snr = calculate_snr(image, signal_roi, noise_roi)
print(f"snr: {snr}")


# In[9]:


snr_list = []

for i in range(len(image_list)):
    IMAGE_NUM = image_list[i]  # 0~34

    image_path = os.path.join(PATH, IMAGE_NUM)
    nifti = nib.load(image_path)
    image = nifti.get_fdata()

    snr = calculate_snr(image, signal_roi, noise_roi)
    print(f"snr {i}: {snr}")
    snr_list.append(snr)


# In[10]:


print(np.min(snr_list), np.max(snr_list), np.mean(snr_list), np.std(snr_list))


# In[ ]:




