# Heuron Research Internship (2024.01.03 - 2024.01.26)


The fMRI dataset used in this project is the company's proprietary asset and cannot be disclosed due to concerns about privacy infringement. Therefore, only the code I have written can be made public.


## 📌 Overview
I worked with the **Parkinson’s disease diagnosis team** at **Heuron**, a company dedicated to leveraging AI technology for neurological disorder solutions, focusing on **medical imaging and deep learning analysis**. During my research and development internship at **Heuron**, I studied basic deep learning models used in medical AI, such as **UNet** and **ResNet50** using **PyTorch**, and further applied these basic model structures to implement **Noise2Noise** and **Noise2Void** for denoising tasks on fMRI images. Additionally, I gained insights into medical imaging, particularly **fMRI** processing, and conducted research on **denoising fMRI images**.

[Heuron](https://iheuron.com/en)

## 🔍 Key Learnings
**Medical Imaging Process:**
  - **Anatomical Position:** Explored **3D planes** and directional terms.
  - **Medical Imaging Formats:** Worked with **DICOM** and **NIfTI** formats.
  - **Coordinate System of DICOM:** Studied how **DICOM** images are structured and referenced.
  - **fMRI:** Gained insights into **magnetic resonance imaging (MRI) techniques**.
  - **SMWI for Parkinson's Disease:** Focused on **Substantia Nigra in the midbrain** using **Susceptibility-Weighted Imaging (SMWI)**.
  - **Medical Data Processing with Python:** Applied **Python** for handling and analyzing medical imaging data.
  - **Noise in fMRI:** Explored various noise types in **fMRI**.

## 🏗 Research Focus: fMRI Denoising
### 🔹 Motivation
- Diseases such as **cerebral hemorrhage** and **Parkinson's disease** require timely treatment within the **golden hour**, but the current medical system often fails to meet this critical window.
- As a result, the importance of **deep-learning based models** that can rapidly diagnose potential diseases using **fMRI images** is increasing.
- However, **fMRI images** are prone to **noise** due to various factors, which negatively impact model performance. Therefore, the **denoising** process is crucial.
- **Medical images lack clean targets**, making traditional **supervised-learning based denoising models** unsuitable.
- **Self-Supervised Learning (SSL)** techniques, such as **Noise2Void**, are commonly used for denoising.

### 🔹 Approach
1. **Implemented Noise2Void** for **fMRI** image denoising.
2. **Trained models** and performed **quantitative evaluations** using **Signal-to-Noise Ratio (SNR)**.
3. **Applied various regularization techniques** to enhance performance.
4. **Compared experimental results** to assess improvements.

## 📊 Results
- Achieved **significant performance improvement** in fMRI denoising.
- Demonstrated the effectiveness of **self-supervised learning** in medical image denoising.

## Discussions
- Despite a significant improvement in **SNR**, a **blurry image** was obtained. A **loss function** design is needed to alleviate this issue.
- Additionally, it may be necessary to explore the use of **Transformer-based models** instead of CNN.
- The Noise2Void model assumes that noise is **independent** of the signal. How should we approach the situation where noise is **not independent** of the signal?

## 🛠 Technologies Used
- **PyTorch**
- **Python**
- **DICOM / NIfTI Processing**
- **Noise2Void**

## 📎 References
- [Nosie2Noise: Learning Image Restoration without Clean Data](https://arxiv.org/pdf/1803.04189)
- [Noise2Void: Learning Denoising from Single Noisy Images](https://arxiv.org/abs/1811.10980)
- [UNet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597)
- [Signal to Nosie ratio in MRI](https://www.researchgate.net/profile/Thomas-Redpath/publication/13515565_Signal-to-noise_ratio_in_MRI/links/0deec529374f34e76a000000/Signal-to-noise-ratio-in-MRI.pdf)
- [Understanding fMRI Noise](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3960487/)

## 📢 Acknowledgments
Special thanks to **Heuron** for the opportunity to work on this exciting research project!
