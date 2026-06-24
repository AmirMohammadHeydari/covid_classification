import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CovidDataset
from models import get_model
from train import train_one_epoch
from eval import evaluate
import config

import matplotlib.pyplot as plt

# dataset
dataset = CovidDataset(config.CT_PATH, config.XRAY_PATH)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=config.LR)

# history
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# training loop
for epoch in range(config.EPOCHS):

    tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

    train_losses.append(tr_loss)
    val_losses.append(val_loss)
    train_accs.append(tr_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # save best model
    if epoch == 0 or val_loss < min(val_losses[:-1]):
        torch.save(model.state_dict(), f"{config.Model_saave_path}/best_model.pt")

plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss Curve")
plt.legend()

# Accuracy
plt.subplot(1,2,2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.title("Accuracy Curve")
plt.legend()

plt.show()