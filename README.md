# 🩺 COVID-19 CT & X-Ray Image Classification using PyTorch

A deep learning project for automatic COVID-19 diagnosis from **CT scan** and **Chest X-ray** images using **Transfer Learning** with **ResNet-50**. The project follows a modular PyTorch architecture and includes data preprocessing, model training, evaluation, and visualization.

---

# 📖 Project Overview

Early detection of COVID-19 from medical images can assist healthcare professionals in making faster clinical decisions. This project implements a complete deep learning pipeline for binary image classification using transfer learning.

The model leverages a pretrained **ResNet-50** backbone and is fine-tuned to distinguish between **COVID-19** and **Normal** cases from both CT scans and chest X-ray images.

---

# ✨ Features

- Modular PyTorch project structure
- Transfer Learning with pretrained ResNet-50
- Multi-modal medical image classification (CT & X-ray)
- Data augmentation and preprocessing
- Training, validation, and testing pipelines
- Automatic best model checkpointing
- Early stopping
- Learning curve visualization
- Confusion Matrix generation
- Classification Report
- Precision, Recall, F1-score, and Accuracy evaluation

---

# 📂 Dataset Structure

```
data/
├── ct/
│   ├── covid/
│   └── normal/
│
└── xray/
    ├── covid/
    └── normal/
```

**Dataset Source**

- Kaggle COVID-19 CT & Chest X-ray Dataset

---

# 🧠 Model Architecture

**Backbone**

- ResNet-50 (ImageNet pretrained)

**Classification Head**

```
Input Image
      │
      ▼
ResNet-50 Backbone
      │
      ▼
Global Average Pooling
      │
      ▼
Linear Layer
      │
      ▼
ReLU
      │
      ▼
Dropout
      │
      ▼
Linear Layer
      │
      ▼
Sigmoid
```

---

# ⚙️ Training Pipeline

The training workflow consists of:

1. Dataset Loading
2. Image Transformations
3. Data Augmentation
4. Train / Validation Split
5. Transfer Learning
6. Binary Classification Training
7. Early Stopping
8. Best Model Saving
9. Performance Evaluation

---

# 📊 Evaluation Metrics

The trained model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

---

# 📈 Training Outputs

During training, the project automatically generates:

- Training Loss Curve
- Validation Loss Curve
- Accuracy Curve
- Confusion Matrix
- Classification Report
- Best Model Checkpoint

---

# 📁 Project Structure

```
covid-classification/

├── data/
├── models/
├── checkpoints/
├── results/
│
├── train.py
├── evaluate.py
├── predict.py
├── config.py
│
└── src/
    ├── dataset.py
    ├── model.py
    ├── trainer.py
    ├── evaluator.py
    ├── transforms.py
    └── utils.py
```

---

# 🚀 Installation

```bash
git clone https://github.com/your-username/covid-classification.git

cd covid-classification

pip install -r requirements.txt
```

---

# ▶️ Training

```bash
python train.py
```

---

# 📊 Evaluation

```bash
python evaluate.py
```

---

# 🔍 Prediction

```bash
python predict.py --image sample.jpg
```

---

# 🛠️ Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- OpenCV
- Matplotlib
- Scikit-Learn

---

# 🔬 Future Improvements

- Vision Transformers (ViT)
- EfficientNet
- DenseNet
- Grad-CAM Visualization
- TensorBoard Integration
- Mixed Precision Training
- Hyperparameter Optimization
- Docker Deployment

---

# 📄 License

This project is intended for educational and research purposes.

