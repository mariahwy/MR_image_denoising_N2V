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


# ## patch

# In[27]:


class NiftiDataset(Dataset):
    def __init__(self, folder_path, transform = None):
        self.forlder_path = folder_path
        self.img_list = sampler(folder_path)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = self.img_list[idx]

        if self.transform:
            sample = self.transform(image)
        
        sample = min_max_normalize(sample)  
        
        return sample

