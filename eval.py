import torch

def evaluate(model, loader, loss_fn, device):

    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for x, y in loader:

            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y.unsqueeze(1).float())

            total_loss += loss.item() * x.size(0)
            correct += (pred.round() == y.unsqueeze(1)).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)