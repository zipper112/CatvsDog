import torch
from torch import nn
import config

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1, padding=0)
        self.cv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.maxpooling = nn.MaxPool2d(kernel_size=2)

        self.acFun = nn.functional.relu
        self.liner1 = nn.Linear(9680, 80)
        self.liner2 = nn.Linear(80, 2)
    def forward(self, x):
        x = self.acFun(self.cv1(x))
        x = self.maxpooling(x)
        x = self.acFun(self.cv2(x))
        x = self.maxpooling(x)
        
        x = x.view(x.shape[0], -1)
        x = self.acFun(self.liner1(x))
        x = self.acFun(self.liner2(x))
        return x
