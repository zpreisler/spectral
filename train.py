#!/usr/bin/env python
import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader

#from matplotlib.pyplot import show,figure,imshow,draw,ion,pause,subplots,subplots_adjust
from numpy import log,array,asarray,save

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
    parser.add_argument('config_file')
    parser.add_argument('-n','--name',default=None)
    parser.add_argument('-l','--learning_rate',default=None,type=float)
    args = parser.parse_args()

    with open(args.config_file,'r') as file:
        config = yaml.load(file,Loader=yaml.FullLoader)

    print('config:',config)
    print('args:',args)

    for k,v in args.__dict__.items():
        if v: config[k] = v

    print('updated_config:',config)

    try:
        os.mkdir(config['name'])
    except OSError as error:
        print(error)

    with open(config['name'] + '/config.yaml', 'w') as file:
        yaml.dump(config,file)
    
    train_data = SpectraDataset(config['data'])
    train_dataloader = DataLoader(train_data,batch_size = config['batch_size'],shuffle=True)
    
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

    current_epoch = 1
    checkpoints = sorted(glob(config['name'] + '/*.pth'))
    if checkpoints:
        print('Loading Last Checkpoint:',checkpoints[-1])
        checkpoint = torch.load(checkpoints[-1])

        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch'] + 1
    
    for epoch in range(current_epoch,config['n_epochs']):
        print('epoch:',epoch)

        for j,_batch in enumerate(train_dataloader):

            inputs,targets = _batch

            _inputs = inputs.to(device)
            _targets = targets.to(device)

            outputs = model(_inputs)

            loss = criterion(outputs,_targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 1 == 0:
            path = config['name'] + '/%04d.pth'%epoch

            print('saving:',path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'loss': loss},
                path)

if __name__ == '__main__':
    main()
