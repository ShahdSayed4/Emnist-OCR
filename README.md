# 🧠 EMNIST OCR - Handwritten Letter Recognition

An AI-based Optical Character Recognition (OCR) system for recognizing
handwritten English letters using classical machine learning techniques.

## 📌 Overview

This project focuses on building an OCR system using the EMNIST dataset
to classify handwritten English alphabet characters.

Unlike deep learning approaches, this project uses **classical machine learning models** combined with feature engineering techniques to achieve high accuracy with better interpretability and lower computational cost. :contentReference[oaicite:0]{index=0}


## 🎯 Objectives

- Recognize handwritten English letters (A–Z)
- Compare performance of:
  - Decision Tree
  - Random Forest
- Evaluate feature extraction techniques:
  - HOG (Histogram of Oriented Gradients)
  - PCA (Principal Component Analysis)
- Build a simple UI for real-time predictions

---

## 🚀 Features

- Upload handwritten letter images
- Predict the letter instantly
- Display prediction results clearly
- Compare model performance

The system allows users to upload an image, click predict, and receive the classified letter. :contentReference[oaicite:1]{index=1}

---
## 🧠 Tech Stack

- Python
- scikit-learn
- NumPy & Pandas
- Matplotlib & Seaborn
- scikit-image (HOG)
- Joblib (model saving/loading)
- Jupyter Notebook

The project uses these tools for preprocessing, modeling, and evaluation. :contentReference[oaicite:2]{index=2}

---

## 🗂️ Dataset

- Dataset: **EMNIST Letters**
- 26 classes (A–Z)
- Grayscale images (28×28 pixels)

Preprocessing steps included:
- Normalization
- Reshaping
- Image rotation & flipping (to fix orientation)
- Train/Validation split

:contentReference[oaicite:3]{index=3}

---
## ⚙️ Methodology

### 🔹 Feature Extraction

#### HOG
- Captures edges and shapes
- Best for handwritten recognition

#### PCA
- Reduces dimensionality
- Faster but may lose details

---

### 🔹 Models Used

#### Decision Tree
- Simple and interpretable
- Prone to overfitting

#### Random Forest
- Ensemble of decision trees
- Better generalization and accuracy

---

## 📊 Results

| Model | Feature | Accuracy |
|------|--------|--------|
| Decision Tree | HOG | ~66% |
| Decision Tree | PCA | ~59% |
| Random Forest | HOG | ~88% |
| Random Forest | PCA | ~82% |
| Random Forest | HOG + PCA | ~86% |

Random Forest significantly outperformed Decision Trees due to ensemble learning. :contentReference[oaicite:4]{index=4}

---

## 🏆 Best Model

👉 **Random Forest + HOG**
- Accuracy ≈ **88.8%**
- Most stable and robust performance

HOG proved better than PCA as it preserves edge and shape information. :contentReference[oaicite:5]{index=5}

---

## 🔄 System Pipeline

1. Input Image
2. Preprocessing (normalize + reshape + rotate)
3. Feature Extraction (HOG / PCA)
4. Model Prediction (DT / RF)
5. Output Predicted Letter

(See system diagrams in report pages 6–8)

---
## ▶️ How to Run

```bash
git clone https://github.com/ShahdSayed4/emnist-ocr.git
cd emnist-ocr
pip install -r requirements.txt
jupyter notebook
```

## 🚀 Run on Google Colab

👉 Click the button below to run the project:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SGPPUfMQlDdpgdswIYycMCk-PUAmA_WM?usp=sharing)


## 👥 Team

- Shahd Sayed 
- Rahma Shaaban
- Kareem Ayman
- Ismael Mahmoud
- Habiba Khaled
- Bavly Aziz

✨ Built with passion for AI
