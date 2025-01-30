from torch import nn


class MLP(nn.Module):

    def __init__(self, max_len=35, dim=25):
        super(MLP, self).__init__()
        self.max_len = max_len
        self.dim = dim
        self.input = nn.Sequential(
            nn.Linear(max_len * dim, int(max_len * dim / 25)),
            nn.ReLU()
        )
        self.hidden = nn.Sequential(
            nn.Linear(int(max_len * dim / 25), max_len),
            nn.ReLU()
        )
        self.output = nn.Linear(max_len, 3)

    def forward(self, x):
        x = x.view(-1, self.max_len * self.dim)
        x = self.input(x)
        x = self.hidden(x)
        return self.output(x)



