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

from glob import glob

def main():
    """
    Main
    """
    parser = ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('-n','--name',default=None)
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

    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    #for var_name in optimizer.state_dict():
    #    print(var_name, "\t", optimizer.state_dict()[var_name])

    checkpoint_name = config['checkpoint_name']
    checkpoint_dir = config['checkpoint_dir']

    #checkpoint = torch.load(checkpoint_dir + checkpoint_name)
    #model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(checkpoint_dir)
    names = sorted(glob(checkpoint_dir + '*_?_*.pth'),key = lambda x: int(x[x.rfind('_') + 1 : x.rfind('.')]) )

    for i,name in enumerate(names):
        outn = checkpoint_dir + '%02d'%i + '.jpeg'
        out = name[:name.rfind('.')] + '.jpeg'
        print(name,out,outn)

        checkpoint = torch.load(name)
        model.load_state_dict(checkpoint['model_state_dict'])
        x = model(train_data.inputs,train_data.log_inputs)

        fig,ax = plot_targets(train_data.targets)
        plot_outputs(ax,x.detach().numpy())
        fig.savefig(outn)
    

if __name__ == '__main__':
    main()
