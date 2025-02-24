#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
import nibabel as nib
import torch

def generate_random_roi(image_shape, num_samples=10, patch_size=20):
    rois = []
    for _ in range(num_samples):
        x = np.random.randint(0, image_shape[0] - patch_size)
        y = np.random.randint(0, image_shape[1] - patch_size)
        rois.append([x, y])
    return rois

def calculate_snr(image_num, num_samples=10, patch_size=20):
    snr_sum = 0

    for i in range(32):  # 32개의 슬라이스에 대해 반복
        image = image_num[:, :, i]

        # 랜덤한 signal & noise ROI 생성
        signal_rois = generate_random_roi(image.shape, num_samples, patch_size)
        noise_rois = generate_random_roi(image.shape, num_samples, patch_size)

        signal_sum = 0
        noise_sum = 0

        for j in range(num_samples):
            signal = image[signal_rois[j][0]:signal_rois[j][0]+patch_size,
                           signal_rois[j][1]:signal_rois[j][1]+patch_size]
            noise = image[noise_rois[j][0]:noise_rois[j][0]+patch_size,
                          noise_rois[j][1]:noise_rois[j][1]+patch_size]

            signal_mean = np.mean(signal)
            noise_std = np.std(noise)

            signal_sum += signal_mean
            noise_sum += noise_std

        mean_signal = signal_sum / num_samples
        std_noise = noise_sum / num_samples

        snr = mean_signal / std_noise if std_noise != 0 else 0 
        snr_sum += snr

    avg_snr = snr_sum / 32
    return avg_snr

# test
PATH = "C:/Users/user/dicom/model/model_5/output"
image_list = os.listdir(PATH)

snr_list = []
for i, image_name in enumerate(image_list):
    image_path = os.path.join(PATH, image_name)
    nifti = nib.load(image_path)
    image = nifti.get_fdata()

    snr = calculate_snr(image, num_samples=10) 
    print(f"SNR {i}: {snr:.4f}")
    snr_list.append(snr)

# SNR 통계 출력
print(f"Min: {np.min(snr_list):.4f}, Max: {np.max(snr_list):.4f}, Mean: {np.mean(snr_list):.4f}, Std: {np.std(snr_list):.4f}")
