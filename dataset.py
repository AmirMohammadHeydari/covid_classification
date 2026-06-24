import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

class CovidDataset(Dataset):

    def __init__(self, ct_path, xray_path):
        self.images = []
        self.labels = []

        # CT COVID = 1
        for img in os.listdir(os.path.join(ct_path, "covid")):
            path = os.path.join(ct_path, "covid", img)
            self.images.append(path)
            self.labels.append(1)

        # CT NORMAL = 0
        for img in os.listdir(os.path.join(ct_path, "normal")):
            path = os.path.join(ct_path, "normal", img)
            self.images.append(path)
            self.labels.append(0)

        # XRAY COVID = 1
        for img in os.listdir(os.path.join(xray_path, "covid")):
            path = os.path.join(xray_path, "covid", img)
            self.images.append(path)
            self.labels.append(1)

        # XRAY NORMAL = 0
        for img in os.listdir(os.path.join(xray_path, "normal")):
            path = os.path.join(xray_path, "normal", img)
            self.images.append(path)
            self.labels.append(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = transform(img)

        label = torch.tensor(self.labels[idx])

        return img, label