#!/usr/bin/env python
import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam

from matplotlib.pyplot import show,figure,imshow,draw,ion,pause,subplots,subplots_adjust
from numpy import log,array,asarray,save

from src.nets import CNNSkip
from src.utils import plot_outputs, plot_targets
from src.dataset import SpectraDataset

from argparse import ArgumentParser
from pathlib import Path

import h5py,yaml
import gc

def main():
    """
    Main
    """

    parser = ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('-n','--name',default=None)
    args = parser.parse_args()

    with open(args.config_file,'r') as file:
        config = yaml.load(file,Loader=yaml.FullLoader)

    print('config:',config)
    print('args:',args)

    if args.name:
        name = args.name
    
    train_data = SpectraDataset(config['data'])
    train_dataloader = DataLoader(train_data,batch_size = config['batch_size'],shuffle=True)

    inputs,targets = next(iter(train_dataloader))
    print(inputs.shape)
    print(targets.shape)
    
    torch.set_num_threads(config['num_threads'])

    model = globals()[config['model']](channels = config['channels'])

    optimizer = optim.SGD(model.parameters(), lr = config['learning_rate'])
    criterion = getattr(nn,config['loss'])()

    if torch.cuda.is_available():
        device = torch.device(config['device'])
    else:
        device = torch.device('cpu')

    model.train()
    model.to(device)

    #checkpoint = torch.load()
    #model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    for epoch in range(config['n_epochs']):
        for j,_batch in enumerate(train_dataloader):

            inputs,targets = _batch

            _inputs = inputs.to(device)
            _targets = targets.to(device)

            outputs = model(_inputs)

            loss = criterion(outputs,_targets)
            print(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


if __name__ == '__main__':
    main()
