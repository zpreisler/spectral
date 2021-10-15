#!/usr/bin/env python
import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader

#from matplotlib.pyplot import show,figure,imshow,draw,ion,pause,subplots,subplots_adjust
from numpy import log,array,asarray,save,vstack

from src.nets import CNNSkip
from src.utils import plot_outputs, plot_targets
from src.dataset import SpectraDataset

from argparse import ArgumentParser
from pathlib import Path
from glob import glob

import h5py,yaml
import gc
import os

def main():
    """
    Main
    """

    parser = ArgumentParser()
    parser.add_argument('checkpoint')
    args = parser.parse_args()

    name = args.checkpoint
    name = name[:name.rfind('.')]

    print('args:',args)
    print('name:',name)

    checkpoint = torch.load(args.checkpoint)

    config = checkpoint['config'] 

    print('config:',config)
    print('loss:',checkpoint['loss'])
    print('epoch:',checkpoint['epoch'])

    train_data = SpectraDataset('../' + config['data'])
    torch.set_num_threads(config['num_threads'])

    print('Load model')
    model = globals()[config['model']](channels = config['channels'])
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)

    if torch.cuda.is_available():
        device = torch.device(config['device'])
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')

    model.eval()
    model.to(device)

    print('Exec model')
    train_dataloader = DataLoader(train_data,batch_size = config['batch_size'],shuffle=False)

    a = []
    for i,batch in enumerate(train_dataloader):

        torch.cuda.empty_cache()

        inputs,targets = batch
        outputs = model(inputs.to(device))

        x = outputs.cpu().detach().numpy()
        a += [x]

        print(i,x.shape)

    a = vstack(a)
    a = a.reshape(train_data.shape)
    b = train_data.targets.reshape(train_data.shape)

    path = name + '.h5'
    print(path)
    with h5py.File(path,'w') as f:
        f.create_dataset('outputs',data = a)
        f.create_dataset('targets',data = b)

if __name__ == '__main__':
    main()
