from torch import nn


class MLP(nn.Module):

    def __init__(self, input_dim=60, output_dim=10, hidden_dim=128):
        super(MLP, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self._net(x)
