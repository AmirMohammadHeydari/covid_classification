import torch

def train_one_epoch(model, loader, loss_fn, optimizer, device):

    model.train()
    total_loss = 0
    correct = 0

    for x, y in loader:

        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y.unsqueeze(1).float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (pred.round() == y.unsqueeze(1)).sum().item()

    return total_loss, correct