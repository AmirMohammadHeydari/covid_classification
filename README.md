🧠 COVID-19 CT & X-Ray Image Classification using PyTorch
📌 Overview

This project is a deep learning pipeline for classifying COVID-19 and normal cases using CT scan and X-ray images.

The model is built using Transfer Learning (ResNet-50) in PyTorch and trained on a mixed dataset of CT and X-ray images.

🚀 Features
🧠 Transfer Learning with ResNet-50
🏥 Multi-modal dataset (CT + X-ray)
📊 Training / Validation / Testing pipeline
📉 Loss & Accuracy visualization
📈 Confusion Matrix & Classification Report
🧱 Fully modular PyTorch code structure
🗂 Dataset

The dataset is structured as:

data/
├── ct/
│   ├── covid/
│   └── normal/
└── xray/
    ├── covid/
    └── normal/

Source: Kaggle COVID-19 X-ray & CT dataset

🧠 Model Architecture
Backbone: ResNet-50 (pretrained on ImageNet)
Fully connected head:
Linear → ReLU → Dropout → Linear → Sigmoid
⚙️ Training Pipeline

The pipeline includes:

Dataset loading using custom PyTorch Dataset
Train / Validation split
Binary Cross Entropy Loss
Adam optimizer
Early stopping (best model saving)
📊 Results

Model performance is evaluated using:

Accuracy
Loss curves
Confusion Matrix
Classification Report
📈 Example Outputs
Training vs Validation loss curve
Accuracy improvement over epochs
Confusion matrix on test set
🧪 Evaluation Metrics
Precision
Recall
F1-score
Accuracy
🧱 Project Structure
dataset.py      → Custom PyTorch Dataset
models/         → ResNet model definition
train.py        → Training loop
eval.py         → Validation loop
test.py         → Final evaluation
main.py         → Entry point
config.py       → Hyperparameters
▶️ How to Run
pip install -r requirements.txt
python main.py
🧠 Key Learnings
Transfer learning with CNNs
Handling multi-class image datasets
Modular deep learning pipeline design
Evaluation metrics in classification tasks
👨‍💻 Author

Student project for learning deep learning with PyTorch.
