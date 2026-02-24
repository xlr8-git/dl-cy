# MRI / X-Ray Enhancer

An AI-powered medical image processing web application built with Streamlit and TensorFlow.

The system automatically classifies uploaded medical images as MRI or X-Ray, enhances their resolution using ESPCN super-resolution models, and highlights the relevant anatomical region.

---

## Overview

This project integrates deep learning classification and super-resolution techniques into a single interactive web application.

### The application performs the following tasks:

- Classifies medical images as **MRI** or **X-Ray**
- Applies **4× super-resolution enhancement** using ESPCN
- Highlights the detected anatomical region
- Displays patient information in a structured interface

---

## Features

### 1. Medical Image Classification

- Binary classification model: `mri_xray_classifier.h5`
- Predicts:
  - MRI
  - X-Ray
- Displays confidence score for prediction

### 2. Super-Resolution Enhancement

- Uses **Efficient Sub-Pixel Convolutional Network (ESPCN)**
- Separate trained models:
  - `best_espcn_model1.keras` (X-Ray)
  - `best_espcn_model2 (1).keras` (MRI)
- 4× image upscaling
- Custom `PixelShuffle` layer implemented

### 3. Anatomical Region Highlighting

- MRI → Head region  
- X-Ray → Chest region  
- Transparent overlay rendering on skeletal template  

---

## Technology Stack

- Python 3.10+
- Streamlit
- TensorFlow
- Keras
- NumPy
- Pillow
- OpenCV

---

## Project Structure

```
dl-cy/
│
├── app.py
├── mri_xray_classifier.h5
├── best_espcn_model1.keras
├── best_espcn_model2 (1).keras
├── human-skeletal-system.jpg
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/xlr8-git/dl-cy.git
cd dl-cy
```

### 2. Create Virtual Environment (Recommended)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If installing manually:

```bash
pip install streamlit tensorflow numpy keras pillow opencv-python
```

---

## Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

## Model Details

### Classification Model

- Input size: 224 × 224 (RGB)
- Output: Binary (MRI / X-Ray)
- Format: `.h5`

### Super-Resolution Models

- ESPCN architecture
- 4× upscaling
- Custom `PixelShuffle` layer
- Format: `.keras`

---

## Deployment Notes

Large model files are managed using Git LFS.

Ensure Git LFS is installed before cloning:

```bash
git lfs install
```

---

## Future Improvements

- Region-based segmentation instead of fixed overlay
- Support for additional medical imaging modalities
- GAN-based super-resolution integration (e.g., Real-ESRGAN)
- Model quantization for faster inference
