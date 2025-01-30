from torch import nn


class MLP(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.input = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, num_classes)

    def forward(self, x0):
        x0 = x0.view(-1, 28 * 28)
        x1 = self.input(x0)
        x2 = self.relu(x1)
        x3 = self.output(x2)
        return x3
