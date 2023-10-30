import torch.nn as nn

class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3,3)),
            nn.ReLU(),
            nn.Conv2d(8, 8, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4608, 10)
        )
    def forward(self, x):
        return self.model(x)