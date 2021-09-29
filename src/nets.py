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
                nn.Conv1d(in_channels=in_channels, out_channels=in_channels, padding=2, kernel_size=5),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Conv1d(in_channels=in_channels, out_channels=in_channels, padding=2, kernel_size=5)
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

class Rugged(nn.Module):
    def __init__(self):
        super().__init__()

        self.l0 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, padding=2, kernel_size=5)
        )

        self.skip_0 = Skip(8,8)
        self.skip_1 = Skip(8,16)
        self.skip_2 = Skip(16,1)

        self.fc = nn.Sequential(
            nn.Linear(512,4),
            nn.Sigmoid()
        )

    def forward(self,x,log_x):

        x = self.l0(x)
        log_x = self.l0(log_x)

        cat_x = torch.cat((x,log_x),2) 

        x = self.skip_0(cat_x)
        x = self.skip_1(x)
        x = self.skip_2(x)

        z = x.flatten(1)
        #print(z.shape)

        z = self.fc(z)

        return z


