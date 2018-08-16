"""
This script generates plots comparing the Pareto fronts of the U- and Cf-switch models for
multiple stress pulse shapes. The data files (in .npy format) should be 
supplied as list of stings containing the path to the data files. Data files of the correct format
are generated from either the plat_* scripts or the gridsearch_* scripts that calculate Pareto fronts.
Additionally, the scripts plots the ratio of the areas under the Pareto fronts for the two switch 
mechanisms for a range of stress magnitudes.

Written by Wylie Stroberg in 2018
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from gridsearch_compare_switches import integrate_pareto_fronts
import os

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
if __name__=="__main__":

    totaluin = 1.0
    basedir = "Data/bruteforce/m100_uc100/totaluin_{}/".format(totaluin)

    # Import data for U-switch
    dU = []
    taupU = []
    Pareto_costsU = []	# list of pareto_points_costs arrays
    for filename in glob.iglob(basedir+"grid_pareto_costsU*"):
        head, tail = os.path.split(filename)
        ext_removed = ('.').join(tail.split('.')[:-1])
        split_name = ext_removed.split("_")
        dU.append(float(split_name[6]))
        taupU.append(float(split_name[8]))
        Pareto_costsU.append(np.load(filename))


    # Import data for Cf-switch
    dC = []
    taupC = []
    Pareto_costsC = []
    for filename in glob.iglob(basedir+"grid_pareto_costsC*"):
        head, tail = os.path.split(filename)
        ext_removed = ('.').join(tail.split('.')[:-1])
        split_name = ext_removed.split("_")
        dC.append(float(split_name[6]))
        taupC.append(float(split_name[8]))
        Pareto_costsC.append(np.load(filename))

    # Sort arrays in order of d
    dU_sorted = sorted(dU)
    taupU_sorted = [x for _,x in sorted(zip(dU,taupU))]
    Pareto_costsU_sorted = [x for _,x in sorted(zip(dU,Pareto_costsU))]

    dC_sorted = sorted(dC)
    taupC_sorted = [x for _,x in sorted(zip(dC,taupC))]
    Pareto_costsC_sorted = [x for _,x in sorted(zip(dC,Pareto_costsC))]


    # Calculate integrals under pareto fronts
    IU = []
    IC = []
    for pU,pC in zip(Pareto_costsU_sorted,Pareto_costsC_sorted):
       print pU
       pcostsU = pU[:,-2:]
       pcostsC = pC[:,-2:]
       print(pcostsU)
       iu,ic,tuncU,truncC = integrate_pareto_fronts(pcostsU,pcostsC)
       #iu,ic,tuncU,truncC = integrate_pareto_fronts(pcostsU-1.,pcostsC-1.)
       IU.append(iu)
       IC.append(ic)

    deltaIUIC = [iu-ic for (iu,ic) in zip(IU,IC)]
    #normedDeltaIUIC = [(iu-ic)/iu for (iu,ic) in zip(IU,IC)]
    normedDeltaIUIC = [iu/ic for (iu,ic) in zip(IU,IC)]
    print(dU_sorted)
    print(dC_sorted)
    # Plot pareto fronts
    figs = []
    for i,(pU,pC,di,taupi) in enumerate(zip(Pareto_costsU_sorted,Pareto_costsC_sorted,dU_sorted,taupU_sorted)):
        pcostsU = pU[:,-2:]
        pcostsC = pC[:,-2:]
        figs.append(plt.subplots(1,1))
        ax = figs[-1][1]
        ax.scatter(pcostsU[:,0],pcostsU[:,1],marker='o',label=r'$U$-switch',color='blue')
        ax.scatter(pcostsC[:,0],pcostsC[:,1],marker='s',label=r'$C_{F}$-switch',color='red')

        ax.set_xlabel("$\mu_{C}$",fontsize=18)
        ax.set_ylabel("$\mu_{U}$",fontsize=18)
        #ax.set_title(r"$d = {}, \tau_{{p}} = {}$".format(di,taupi))
        #ax.text(0.5,0.9,r"$d = {:.2f}, \tau_{{p}} = {:.2f}$".format(di,taupi),fontsize=16,ha='center',transform=ax.transAxes)
        ax.text(0.5,0.9,r"$D = {:.2f}$".format(di),fontsize=16,ha='center',transform=ax.transAxes)
        ax.text(0.5,0.83,r"$\tau_{{p}} = {:.2f}$".format(taupi),fontsize=16,ha='center',transform=ax.transAxes)
        if i==0:
            ax.legend()

        ax.locator_params(axis='both',nbins=5)
        ax.tick_params(axis='both',labelsize=16)

        figs[-1][0].tight_layout()
        figs[-1][0].savefig("Figures/Pareto_front_compare_d_{}_taup_{}_totaluin_{}.eps".format(di,taupi,totaluin),edgecolor='black')

    # Plot difference in area under pareto fronts
    fig,ax = plt.subplots(1,1)

    ax.plot(dU_sorted,deltaIUIC,".",markersize=15)
    ax.set_xscale('log')
    ax.set_xlabel('D',fontsize=18)
    ax.set_ylabel(r'$\Delta_{P}$',fontsize=18)

    ax.locator_params(axis='y',nbins=5)
    ax.tick_params(axis='both',labelsize=16)

    fig.tight_layout()
    #fig.savefig("Figures/Delta_IUIC_vs_d_total_uin{}.eps".format(totaluin),edgecolor='black')

    # Plot normalized difference in area under pareto fronts (normalized by mu_u integral)
    fig2,ax2 = plt.subplots(1,1)

    ax2.plot(dU_sorted,normedDeltaIUIC,".",markersize=15)
    ax2.set_xscale('log')
    ax2.set_xlabel('D',fontsize=18)
    ax2.set_ylabel(r'$\Delta_{P}$',fontsize=18)

    ax2.locator_params(axis='y',nbins=5)
    ax2.tick_params(axis='both',labelsize=16)

    fig2.tight_layout()
    fig2.savefig("Figures/Delta_IUIC_vs_d_total_uin{}.eps".format(totaluin),edgecolor='black')


    plt.show()
