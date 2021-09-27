#!/usr/bin/env python
import numpy as np
from numpy import histogram
from matplotlib.pyplot import figure,show,plot,imshow,title,semilogy,semilogx,loglog
from glob import glob
import h5py

input_files = sorted(glob('mockup_XRF/Z0*.edf'))
target_files = sorted(glob('mockup_XRF/i*.edf'))

def read_edf_line(name,n,shape=(-1,2048)):
    with open(name,'rb') as f:
        for _ in range(n):
            f.readline()

        x = np.frombuffer(f.read(),'d')
        x = x.reshape(*shape)

    return x

def read_edf(files,n,shape):
    x = []
    for file in files:
        x += [read_edf_line(file,n,shape)]

    return np.asarray(x)

def main():

    inputs = read_edf(input_files,14,(-1,2048))
    targets = read_edf(target_files,10,(69,94))

    targets = targets.swapaxes(0,1).swapaxes(1,2)

    print(inputs.shape,inputs.max())
    print(targets.shape,targets.max())

    print('Saving data.h5')
    with h5py.File('data.h5','w') as f:
        f.create_dataset('inputs',data = inputs)
        f.create_dataset('targets',data = targets)


    t_ = targets.swapaxes(0,2)
    print(t_.min(),t_.max())

    for i,(t,f) in enumerate(zip(t_,target_files)):
        print(i,f,t.shape,t.min(),t.max(),t.mean())

        t[t<0] = 0
        t = t / (t.max())

        h,b=histogram(t.flatten(),bins=512,range=(0,t.max()))
        figure()
        title(r'%s'%f.replace('_',' '))
        semilogy(b[1:],h)

        figure()
        imshow(t.T)
        title(r'%s'%f.replace('_',' '))

    show()


if __name__ == '__main__':
    main()
