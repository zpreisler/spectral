import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam

from numpy import log
import h5py

class SpectraDataset(Dataset):

    def __init__(self,source):
        inputs,self.targets = self.load_h5(source)

        self.targets = self.targets[:,:,8:12]
        
        self.targets[self.targets < 0] = 0.01
        self.targets = self.targets / self.targets.max()

        self.inputs = inputs / inputs.max()

        inputs[inputs <= 0] = 0.001
        self.log_inputs = log(inputs)
        self.log_inputs = self.log_inputs / self.log_inputs.max()

        self.inputs = self.inputs.reshape((-1,1,self.inputs.shape[-1]))
        self.log_inputs = self.log_inputs.reshape((-1,1,self.log_inputs.shape[-1]))

        self.targets = self.targets.reshape((-1,self.targets.shape[-1]))

        self.inputs = torch.from_numpy(self.inputs).float()
        self.log_inputs = torch.from_numpy(self.log_inputs).float()

        self.targets = torch.from_numpy(self.targets).float()

        print('Inputs',self.inputs.shape)
        print('Targets',self.targets.shape)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,idx):

        inputs = self.inputs[idx]
        log_inputs = self.log_inputs[idx]

        targets = self.targets[idx]

        return inputs,log_inputs,targets

    def load_h5(self,name):
        with h5py.File(name,'r') as f:
            x = f['inputs']
            inputs = x[:]
            y = f['targets']
            targets = y[:]

        return inputs,targets
