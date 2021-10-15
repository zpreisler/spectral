import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam

from numpy import log
import h5py

class SpectraDataset(Dataset):

    def __init__(self,source):
        self.inputs,self.targets = self.load_h5(source)

        self.shape = self.targets.shape

        self.targets = self.targets.reshape((-1,self.targets.shape[-1]))
        self.inputs = self.inputs.reshape((-1,2,self.inputs.shape[-1]))

        self.targets = torch.from_numpy(self.targets).float()
        self.inputs = torch.from_numpy(self.inputs).float()

        print('Inputs:',self.inputs.shape)
        print('Targets:',self.targets.shape)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,idx):

        targets = self.targets[idx]
        inputs = self.inputs[idx]

        return inputs,targets

    def load_h5(self,name):
        with h5py.File(name,'r') as f:
            x = f['inputs']
            inputs = x[:]
            y = f['targets']
            targets = y[:]

        return inputs,targets
