from torch import nn


class CNN21(nn.Module):

    def __init__(self, num_classes=10):
        super(CNN21, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.output = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x0):
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x2_ = x2.view(x2.size(0), -1)
        return self.output(x2_)


class CNN32(nn.Module):

    def __init__(self, num_classes=10):
        super(CNN32, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 3*32*32 -> 16*32*32
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 16*32*32 -> 16*16*16
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),  # 16*16*16 -> 32*16*16
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 32*16*16 -> 32*8*8
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),  # 32*8*8 -> 64*8*8
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 64*8*8 -> 64*4*4
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 32),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(-1, 64 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
