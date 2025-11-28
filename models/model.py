import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten(2) 
        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x=self.conv1(x).relu()
        x=self.conv2(x).relu()
        x=self.conv3(x).relu()
        x=self.conv4(x).relu()
        x=self.conv5(x).relu()
        x=self.flatten(x).mean(-1)
        return self.fc1(x)
