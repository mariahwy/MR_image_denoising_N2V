#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import numpy as np
import pandas as pd


# In[63]:


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv(x)


# In[62]:


class UpConv(nn.Module):
    def __init__(self):
        super(UpConv, self).__init__()

    def forward(self, x):
        channel = x.shape[1]
        upconv = nn.ConvTranspose2d(in_channels = channel, out_channels = int(channel/2), kernel_size = 2, stride = 2, padding = 0)
        x = upconv(x)

        return x


# In[69]:


class CenterCrop(nn.Module):
    def __init__(self, upconv_input):
        super(CenterCrop, self).__init__()
        
        self.centercrop = torchvision.transforms.CenterCrop(upconv_input.shape[3])
    
    def forward(self, y):
        return self.centercrop(y)


# In[71]:


class Cat(nn.Module):
    def __init__(self):
        super(Cat, self).__init__()

    def forward(self, x, y):
        return torch.cat((x, y), dim = 1)


# In[74]:


class ContPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContPath, self).__init__()
        
        self.conv = ConvBlock(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)

        return x


# In[80]:


class ExpPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExpPath, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.upconv = UpConv()
        self.cat = Cat()

    def forward(self, upconv_input, crop_input):
        x = self.upconv(upconv_input)
        centercrop = CenterCrop(x)
        y = centercrop(crop_input)
        result = self.cat(x, y)
        result = self.conv(result)

        return result

