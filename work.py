#!/usr/bin/env python
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam

from matplotlib.pyplot import show,figure,imshow,draw,ion,pause

from numpy import log

import h5py

def load_h5(name):
    with h5py.File(name,'r') as f:
        x = f['inputs']
        inputs = x[:]
        y = f['targets']
        targets = y[:]

    return inputs,targets

class SpectraDataset(Dataset):
    def __init__(self,source):
        self.inputs,self.targets = self.load_h5(source)


        self.targets = self.targets[:,:,8:12]
        
        self.targets[self.targets < 0] = 0.01
        self.targets = self.targets / self.targets.max()

        self.inputs[self.inputs <= 0] = 0.001
        self.inputs = log(self.inputs)
        self.inputs = self.inputs / self.inputs.max()


        self.inputs = self.inputs.reshape((-1,1,self.inputs.shape[-1]))
        self.targets = self.targets.reshape((-1,self.targets.shape[-1]))
        print('targets.shape:',self.targets.shape)

        self.inputs = torch.from_numpy(self.inputs).float()
        self.targets = torch.from_numpy(self.targets).float()

        print('Inputs',self.inputs.shape)
        print('Targets',self.targets.shape)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,idx):
        inputs = self.inputs[idx]
        targets = self.targets[idx]

        return inputs,targets

    def load_h5(self,name):
        with h5py.File(name,'r') as f:
            x = f['inputs']
            inputs = x[:]
            y = f['targets']
            targets = y[:]

        return inputs,targets

class Rugged(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, padding=2, kernel_size=5),

            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.l2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, padding=2, kernel_size=5),

            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.l3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, padding=2, kernel_size=5),

            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.l4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, padding=2, kernel_size=5),

            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.l5 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, padding=2, kernel_size=5),

            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
 
        self.fc = nn.Sequential(
                nn.Linear(2048,4),
                nn.Sigmoid()
                )

    def forward(self,x):

        y = self.l1(x)
        y = self.l2(y)
        y = self.l3(y)
        y = self.l4(y)
        y = self.l5(y)

        z = y.flatten(1)

        z = self.fc(z)

        return z

def main():
    print('main')

    
    train_data = SpectraDataset('data.h5')
    train_dataloader = DataLoader(train_data,batch_size = 64,shuffle=True)

    #inputs,targets = next(iter(train_dataloader))
    #print(inputs,targets,inputs.shape,targets.shape)

    model = Rugged()

    params = list(model.parameters())
    #print(params)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    figure(1)

    figure(2)

    figure(3)

    figure(4)

    figure(5)
    z = train_data.targets[:,0].reshape(69,94)
    imshow(z)

    figure(6)
    z = train_data.targets[:,1].reshape(69,94)
    imshow(z)

    figure(7)
    z = train_data.targets[:,2].reshape(69,94)
    imshow(z)

    figure(8)
    z = train_data.targets[:,3].reshape(69,94)
    imshow(z)

    ion()
    show()

    with open('log','w') as flog:

        for e in range(200):
            for i,batch_ in enumerate(train_dataloader):
                inputs,targets = batch_

                outputs = model(inputs)

                loss = torch.sqrt(criterion(outputs,targets))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


                if i % 300 == 0:

                    x = model(train_data.inputs)

                    z = x.detach()[:,0].numpy().reshape(69,94)
                    figure(1)
                    imshow(z)

                    z = x.detach()[:,1].numpy().reshape(69,94)
                    figure(2)
                    imshow(z)

                    z = x.detach()[:,2].numpy().reshape(69,94)
                    figure(3)
                    imshow(z)

                    z = x.detach()[:,3].numpy().reshape(69,94)
                    figure(4)
                    imshow(z)

                    draw()
                    pause(0.001)

                if i % 100 == 0:

                    print(i,outputs.shape,targets.shape)
                    print(outputs[0],targets[0],targets[0]-outputs[0])

                    print('loss: %s'%loss.detach().numpy())
                    flog.write('%f\n'%loss.detach().numpy())
                    flog.flush()



if __name__ == '__main__':
    main()
