"""
Contains several functions for plotting the activation functions for the
U- and Cf-switches.

Written by Wylie Stroberg in 2018

"""

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
mpl.rc('axes', linewidth=3)
import matplotlib.pyplot as plt

#----------------------------------------------------------------------
def piecewise_linear_Uswitch(x,m):
    if(x<=0.0):
        return 0.0
    elif(x>=1.0/m):
        return 1.0
    else:
        return x*m
#----------------------------------------------------------------------
def piecewise_linear_Cfswitch(x,m):
    if(x>=0.0):
        return 0.0
    elif(x<=-1.0/m):
        return 1.0
    else:
        return -x*m
#----------------------------------------------------------------------
def plot_switch(xcrit,m,**kwargs):#Xrange=(1,10),switch_func=piecewise_linear_Uswitch,axis=None,color='b'):

    # Default kwargs
    X = np.linspace(0,10,1000)
    switch_func=piecewise_linear_Uswitch
    ax=None
    color='blue'

    # Parse kwargs
    if 'x' in kwargs:
        X = kwargs['x']
    if 'switch' in kwargs:
        if kwargs['switch']=='uswitch':
            switch_func = piecewise_linear_Uswitch
        elif kwargs['switch']=='cfswitch':
            switch_func = piecewise_linear_Cfswitch
        else:
            switch_func = kwargs['switch']
    if 'axis' in kwargs:
        ax = kwargs['axis']
    if 'color' in kwargs:
        color = kwargs['color']

    # Plot switch function
    if ax==None:
        fig,ax = plt.subplots(1,1)

    g = [switch_func(Xi-xcrit,m) for Xi in X]
    p = ax.plot(X,g,linewidth=3,color=color)

    #------- Axis and Figure Adjustments ----------#

    #ax.set_xlabel(r'$U/U_{ss}$',fontsize=18)
    #ax.set_ylabel(r'$G\left(U\right))/G_{0}$',fontsize=18)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # To specify the number of ticks on both or any single axes
    ax.locator_params(axis='y', nbins=2)
    ax.locator_params(axis='x', nbins=int(X[-1]))

    # Change strings for y ticklabels
    ax.set_ylim((-0.05,1.05))
    ax.set_yticks((0.,1.))
    ylabel_text = ['' for i in ax.get_yticks()]
    #ylabels = [item.get_text() for item in ax.get_yticklabels()]
    ylabel_text[0] = r"0.0"
    ylabel_text[-1] = r"$G_{0}$"
    #ylabels[1:3] = ['0',r'$G_{0}$']
    #ax.set_yticklabels(ylabels)
    ax.set_yticklabels(ylabel_text)
 
    # Change the fontsize of minor ticks label 
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Change strings for x ticklabels
    ax.set_xticks((X[0],X[-1]))
    label_text = ['' for i in ax.get_xticks()]
    label_text[0] = r"${:.1f}$".format( ax.get_xticks()[0])
    label_text[-1] = r"${:.1f}$".format( ax.get_xticks()[-1])
    ax.set_xticklabels(label_text)


    #fig.tight_layout()
    if 'axis' not in kwargs:
        return fig, ax
    else:
        return
#----------------------------------------------------------------------
#----------------------------------------------------------------------
if __name__=="__main__":
    # switch function parameters
    ubar = 2.#e5
    m = 1.0
    U = np.linspace(0,10,100)
    
    cmax=2.0
    Cf = np.linspace(0,10,100)

    figU, axU = plot_switch(ubar,m,x=U,switch=piecewise_linear_Uswitch)
    figCf, axCf = plot_switch(ubar,m,x=Cf,switch=piecewise_linear_Cfswitch)

    figname = './Figures/switch_functions/piecewise_ubar_{:.1}_m_{:}.eps'.format(ubar,m)
    #fig.savefig(figname)

    plt.show()
