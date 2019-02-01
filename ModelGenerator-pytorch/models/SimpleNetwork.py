from torch import nn
import torch.nn.functional as F


class SimpleNetwork(nn.Module):
    """ A rudimentary configuration for starting """

    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 7, padding=1)
        self.conv2 = nn.Conv2d(64, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 192, 3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(192 * 13 * 13, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self) -> str:
        """ Returns the name of this configuration """
        return "simple"
