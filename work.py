#!/usr/bin/env python
import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam

from matplotlib.pyplot import show,figure,imshow,draw,ion,pause,subplots,subplots_adjust

from numpy import log,array,asarray,save

from src.nets import Rugged
from src.utils import plot_outputs, plot_targets
from src.dataset import SpectraDataset

from pathlib import Path
from argparse import ArgumentParser
import h5py,yaml
import gc

#def load_h5(name):
#    with h5py.File(name,'r') as f:
#        x = f['inputs']
#        inputs = x[:]
#        y = f['targets']
#        targets = y[:]
#
#    return inputs,targets
#
#class SpectraDataset(Dataset):
#
#    def __init__(self,source):
#        inputs,self.targets = self.load_h5(source)
#
#        self.targets = self.targets[:,:,8:12]
#        
#        self.targets[self.targets < 0] = 0.01
#        self.targets = self.targets / self.targets.max()
#
#        self.inputs = inputs / inputs.max()
#
#        inputs[inputs <= 0] = 0.001
#        self.log_inputs = log(inputs)
#        self.log_inputs = self.log_inputs / self.log_inputs.max()
#
#        self.inputs = self.inputs.reshape((-1,1,self.inputs.shape[-1]))
#        self.log_inputs = self.log_inputs.reshape((-1,1,self.log_inputs.shape[-1]))
#
#        self.targets = self.targets.reshape((-1,self.targets.shape[-1]))
#
#        self.inputs = torch.from_numpy(self.inputs).float()
#        self.log_inputs = torch.from_numpy(self.log_inputs).float()
#
#        self.targets = torch.from_numpy(self.targets).float()
#
#        print('Inputs',self.inputs.shape)
#        print('Targets',self.targets.shape)
#
#    def __len__(self):
#        return len(self.inputs)
#
#    def __getitem__(self,idx):
#
#        inputs = self.inputs[idx]
#        log_inputs = self.log_inputs[idx]
#
#        targets = self.targets[idx]
#
#        return inputs,log_inputs,targets
#
#    def load_h5(self,name):
#        with h5py.File(name,'r') as f:
#            x = f['inputs']
#            inputs = x[:]
#            y = f['targets']
#            targets = y[:]
#
#        return inputs,targets
#
#class Skip(nn.Module):
#    def __init__(self,in_channels,out_channels):
#        super().__init__()
#
#        self.conv = nn.Sequential(
#                nn.Conv1d(in_channels=in_channels, out_channels=in_channels, padding=2, kernel_size=5),
#                nn.BatchNorm1d(in_channels),
#                nn.ReLU(),
#                nn.Conv1d(in_channels=in_channels, out_channels=in_channels, padding=2, kernel_size=5)
#                )
#
#        self.pooling = nn.Sequential(
#                nn.BatchNorm1d(in_channels),
#                nn.ReLU(),
#                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, padding=0, kernel_size=1),
#                nn.MaxPool1d(kernel_size=2, stride=2)
#                )
#
#    def forward(self,x):
#        y = self.conv(x)
#        x = x + y
#        x = self.pooling(x)
#        return x
#
#class Rugged(nn.Module):
#    def __init__(self):
#        super().__init__()
#
#        self.l0 = nn.Sequential(
#            nn.Conv1d(in_channels=1, out_channels=8, padding=2, kernel_size=5)
#        )
#
#        self.skip_0 = Skip(8,8)
#        self.skip_1 = Skip(8,16)
#        self.skip_2 = Skip(16,1)
#
#        self.fc = nn.Sequential(
#            nn.Linear(512,4),
#            nn.Sigmoid()
#        )
#
#    def forward(self,x,log_x):
#
#        x = self.l0(x)
#        log_x = self.l0(log_x)
#
#        cat_x = torch.cat((x,log_x),2) 
#
#        x = self.skip_0(cat_x)
#        x = self.skip_1(x)
#        x = self.skip_2(x)
#
#        z = x.flatten(1)
#        #print(z.shape)
#
#        z = self.fc(z)
#
#        return z
#
def main():
    """
    Main
    """

    torch.set_num_threads(1)

    parser = ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('-n','--name',default=None)
    parser.add_argument('-p','--plot',action='store_true')
    parser.add_argument('-q','--quiet',action='store_true')
    args = parser.parse_args()

    with open(args.path,'r') as file:
        config = yaml.load(file,Loader=yaml.FullLoader)

    print(args,config)

    model_name = config['model_name']
    model_dir = config['model_dir']

    if args.name:
        model_name = args.name
    
    training_data_name = config['training_data_name']
    training_data_dir = config['training_data_dir']

    train_data = SpectraDataset(training_data_dir + training_data_name)
    train_dataloader = DataLoader(train_data,batch_size = config['batch_size'],shuffle=True)

    model = Rugged()
    optimizer = optim.SGD(model.parameters(), lr = config['learning_rate'])

    criterion = nn.L1Loss()
    criterion = getattr(nn,config['loss'])()
    #criterion = nn.MSELoss()
    #criterion = nn.HuberLoss(delta=2)
    #criterion = nn.SmoothL1Loss()

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    if 'checkpoint_name' in config:
        checkpoint_name = config['checkpoint_name']
        checkpoint_dir = config['checkpoint_dir']

        checkpoint = torch.load(checkpoint_dir + checkpoint_name)
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    model.train()

    if args.plot:
        ax = plot_targets(train_data.targets)
        ion()
        show()

    print('Number of batches',len(train_dataloader))
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model.to(device)
    
    with open(model_dir + model_name + '_log','w') as f_log:

        for epoch in range(512):
            for j,batch_ in enumerate(train_dataloader):

                inputs,log_inputs,targets = batch_

                outputs = model(inputs,log_inputs)

                loss = criterion(outputs,targets)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                f_log.write('%f\n'%loss.detach().numpy())

            if epoch % 1 == 0:
                #loss_ = loss.detach().numpy()
                print('epoch:',epoch)
                print('loss: ',loss)
                print(outputs[0],targets[0],targets[0]-outputs[0])
                f_log.flush()

                #x = model(train_data.inputs,train_data.log_inputs)
                #with h5py.File(model_dir + 'outputs%d.h5'%epoch,'w') as f_h5:
                #    x = model(train_data.inputs,train_data.log_inputs).detach().cpu().numpy()
                #    f_h5.create_dataset('output',data = x)
                #    del(x)

            if epoch % 4 == 0:
                path = model_dir + model_name + '_%d.pth'%epoch
                print('Checkpoint:',path)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                    },path)

            if args.plot: 
                if epoch % 2 == 0:
                    x = model(train_data.inputs,train_data.log_inputs)
                    plot_outputs(ax,x.detach().numpy())
                    del(x)

            gc.collect()

if __name__ == '__main__':
    main()
