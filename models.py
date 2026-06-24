import torchvision.models as models
import torch.nn as nn

def get_model():

    model = models.resnet50(pretrained=True)

    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

    return model