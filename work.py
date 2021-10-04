#!/usr/bin/env python
import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam

from matplotlib.pyplot import show,figure,imshow,draw,ion,pause,subplots,subplots_adjust

from numpy import log,array,asarray,save

from src.nets import Rugged2
from src.utils import plot_outputs, plot_targets
from src.dataset import SpectraDataset

from pathlib import Path
from argparse import ArgumentParser
import h5py,yaml
import gc

def main():
    """
    Main
    """
    torch.set_num_threads(1)

    parser = ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('-n','--name',default=None)
    parser.add_argument('-w','--width',default=None,type=int)
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
    
    if args.width:
        print('width:',args.width)
        model = Rugged2(w=args.width)
    else:
        model = Rugged2()

    optimizer = optim.SGD(model.parameters(), lr = config['learning_rate'])

    criterion = getattr(nn,config['loss'])()

    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    #for var_name in optimizer.state_dict():
    #    print(var_name, "\t", optimizer.state_dict()[var_name])

    if 'checkpoint_name' in config:
        checkpoint_name = config['checkpoint_name']
        checkpoint_dir = config['checkpoint_dir']

        checkpoint = torch.load(checkpoint_dir + checkpoint_name)
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    if args.plot:
        fig,ax = plot_targets(train_data.targets)
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

            if epoch % 2 == 0:
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

                    draw()
                    pause(0.1)

            gc.collect()

if __name__ == '__main__':
    main()
