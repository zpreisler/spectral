from matplotlib.pyplot import show,figure,imshow,draw,ion,pause,subplots,subplots_adjust

def plot_targets(targets):

    n = targets.shape[-1]
    fig, ax = subplots(2,n,figsize=(n*4,5))

    fig.subplots_adjust(left=0,right=1,top=1,bottom=0,wspace=0.01,hspace=0.01)

    for i in range(n):
        z = targets[:,i].reshape(69,94)
        ax[0,i].imshow(z)
        ax[0,i].set_xticklabels([])
        ax[0,i].set_yticklabels([])

        ax[1,i].imshow(z)
        ax[1,i].set_xticklabels([])
        ax[1,i].set_yticklabels([])

    return fig,ax

def plot_outputs(ax,outputs):

    n = outputs.shape[-1]
    for i in range(n):
        z = outputs[:,i].reshape(69,94)

        ax[1,i].imshow(z)
        ax[1,i].set_xticklabels([])
        ax[1,i].set_yticklabels([])
