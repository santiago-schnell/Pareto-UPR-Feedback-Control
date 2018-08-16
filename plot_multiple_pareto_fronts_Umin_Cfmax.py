"""
This script generates a plot comparing multiple Pareto fronts. The data files (in .npy format) should be 
supplied as list of stings containing the path to the data files. Data files of the correct format
are generated from either the plat_* scripts or the gridsearch_* scripts that calculate Pareto fronts.

Written by Wylie Stroberg in 2018
"""

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
mpl.rc('axes', linewidth=2)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pydelay import dde23
import delayed_nf_upr_piecewise_u_switch as upw
import delayed_nf_upr_piecewise_cf_switch as cpw

#----------------------------------------------------------------------
#----------------------------------------------------------------------
if __name__=="__main__":
   
    # Specifiy paths for input data files

    param_tag = "_1.0_0.5_100000"
    pareto_front_data_file = ['./Data/U_switch/plat/pareto_front'+param_tag+'_NSGAII_dtmax0.003.npy',
                              './Data/Cf_switch/plat/fixedGain/pareto_front'+param_tag+'_NSGAII_dtmax0.003.npy',
                              './Data/AND_switch/plat/fixedGain/pareto_front'+param_tag+'_NSGAII_dtmax0.003_relaxedBounds.npy']


    prob_dims = [(2,2),(2,2),(4,2)] # list of (Nvars,Nobjs) for each model (AND-switch has 4 variables as opposed to 2)

    # Load Pareto front data from data files
    Pareto_points = []
    Pareto_costs = []
    for i,pf in enumerate(pareto_front_data_file):
        pareto_data = np.load(pf)
        nv = prob_dims[i][0]
        Pareto_points.append(pareto_data[:,0:nv])
        Pareto_costs.append(pareto_data[:,nv:])
        print("Parameters at minimum mu_U:")
        print(pareto_data[np.argmin(pareto_data[:,nv+1]),:])

    # Indices for phase-plane plotting (set manually)
    ind_samples = [70,70,65]
    plot_inds = []
    for i,(ind_sample,pareto_costs) in enumerate(zip(ind_samples,Pareto_costs)):
        # indices of pareto-optimal costs sorted by C-cost values
        pcosts_Csorted_inds = sorted(range(len(pareto_costs)), key=lambda k: pareto_costs[k][0])

        # indices of the unsorted arrays corresponding to the sampled, sorted values
        plot_inds.append(pcosts_Csorted_inds[ind_sample])

    ind_samples2 = [22,25,21]
    if ind_samples2 is not None:
        plot_inds2 = []
        for i,(ind_sample,pareto_costs) in enumerate(zip(ind_samples2,Pareto_costs)):
            # indices of pareto-optimal costs sorted by C-cost values
            pcosts_Csorted_inds = sorted(range(len(pareto_costs)), key=lambda k: pareto_costs[k][0])

            # indices of the unsorted arrays corresponding to the sampled, sorted values
            plot_inds2.append(pcosts_Csorted_inds[ind_sample])

    print("********************************************************")
    print(plot_inds)
    print(plot_inds2)
    print("********************************************************")
    #plot_inds = [44,20,18]
    #plot_inds = [None,None,None]

    #-----------------------------------------------------------#
    #----- Plot pareto front -----#
    figP,ax = plt.subplots(1,1,figsize=(8,4))

    labels = [r'$U$-switch',r'$C_F$-switch',r'AND-Switch']
    colors = ['blue','red','green']
    markers = ['o','s','*']

    for i,(pareto_points,pareto_costs) in enumerate(zip(Pareto_points,Pareto_costs)):
        plot_ind = plot_inds[i]
        if ind_samples2 is not None:
            plot_ind2 = plot_inds2[i]
        else:
            plot_ind2 = None
        label = labels[i]
        color = colors[i]
        marker = markers[i]
        ms = 3
        # plot pareto front
        for j in range(pareto_costs.shape[0]):
            if j == plot_ind:
                ax.plot(pareto_costs[j,0],pareto_costs[j,1],
                        linestyle='', marker='^', markeredgecolor='black',
                         markerfacecolor='black', markersize=ms+3)
            if j == plot_ind2:
                ax.plot(pareto_costs[j,0],pareto_costs[j,1],
                        linestyle='', marker='v', markeredgecolor='black',
                         markerfacecolor='black', markersize=ms+3)
 
        ax.scatter(pareto_costs[:,0],pareto_costs[:,1],s=15,color=color,marker=marker,label=label)


    ax.set_xlabel(r'$\mu_{C}$',fontsize=18)
    ax.set_ylabel(r'$\mu_{U}$',fontsize=18)

    ax.locator_params(axis='both', nbins=5)

    ax.set_xlim((0.97,1.23))
    ax.set_ylim(1.15,ax.get_ylim()[1])

    # Optionally draw inset with zoomed view
    draw_inset = False
    if draw_inset:
        # Draw rectange marking inset zoomed view
        rect = patches.Rectangle((1.08,1.28),1.11-1.08,1.34-1.28,linewidth=1,edgecolor='black',facecolor='none')
        ax.add_patch(rect)

        ax.legend(loc=3)

        figP.tight_layout()

        #figP.savefig('./Figures/ParetoFrontCompare_U_Cf'+param_tag+'.eps',edgecolor='black')
        figP.savefig('./Figures/ParetoFrontCompare_U_Cf_AND'+param_tag+'.eps',edgecolor='black')

        # Create zoomed view for inset
        ax.set_xlim((1.08,1.11))
        ax.set_ylim((1.28,1.34))
        ax.legend().remove()
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks(())
        ax.set_yticks(())
        rect.set_visible(False)

        figP.savefig('./Figures/ParetoFrontCompare_U_Cf_AND'+param_tag+'_inset.eps',edgecolor='black')
    else:
        ax.legend()
        figP.tight_layout()

        #figP.savefig('./Figures/ParetoFrontCompare_U_Cf'+param_tag+'.eps',edgecolor='black')
        figP.savefig('./Figures/ParetoFrontCompare_U_Cf_AND'+param_tag+'.eps',edgecolor='black')



    plt.show()
