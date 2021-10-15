import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam

from matplotlib.pyplot import show,figure,imshow,draw,ion,pause,subplots,subplots_adjust
from numpy import log,array,asarray,save

class Skip(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.conv = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=in_channels, padding=2, kernel_size=5,bias=False),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Conv1d(in_channels=in_channels, out_channels=in_channels, padding=2, kernel_size=5,bias=False)
                )

        self.pooling = nn.Sequential(
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, padding=0, kernel_size=1),
                nn.MaxPool1d(kernel_size=2, stride=2)
                )

    def forward(self,x):
        y = self.conv(x)
        x = x + y
        x = self.pooling(x)
        return x

class CNNSkip(nn.Module):
    def __init__(self,channels = 16, kernel_size = 5):
        super().__init__()

        self.l0 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=channels, padding= int(kernel_size / 2), kernel_size=kernel_size)
        )

        self.reduce = nn.MaxPool1d(kernel_size=2, stride=2)

        self.skip_0 = Skip(channels,channels)
        self.skip_1 = Skip(channels,channels)
        self.skip_2 = Skip(channels,channels)
        self.skip_3 = Skip(channels,1)

        self.fc = nn.Sequential(
            nn.Linear(128,4),
            nn.Sigmoid()
        )

    def forward(self,x):

        x = self.l0(x)

        x = self.skip_0(x)
        x = self.skip_1(x) + self.reduce(x)
        x = self.skip_2(x) + self.reduce(x)
        x = self.skip_3(x)

        z = x.flatten(1)
        #print(z.shape)

        z = self.fc(z)

        return z


