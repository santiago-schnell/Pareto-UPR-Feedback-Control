"""
This script generates a plot comparing multiple Pareto fronts and colors the Pareto fronts based
on the maximal unfolded protein level during the simulation timecourse. The data files
(in .npy format) should be supplied as list of stings containing the path to the data files. 
Data files of the correct format are generated from either the plat_* scripts or the gridsearch_* 
scripts that calculate Pareto fronts.

Written by Wylie Stroberg in 2018
"""

#----------------------------------------------------------------------
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
mpl.rc('axes', linewidth=2)
import matplotlib.pyplot as plt
from pydelay import dde23
import delayed_nf_upr_piecewise_u_switch as upw
import delayed_nf_upr_piecewise_cf_switch as cpw

#----------------------------------------------------------------------
#----------------------------------------------------------------------
if __name__=="__main__":
   
    #------------- Choose Data Files for Plotting ------------#

    basedir = './Data/bruteforce/m100_uc100/totaluin_1.0/'
    param_tag = "_totaluin_1.0_d_2.15443469003_taup_0.464158883361"

    pareto_front_data_file = [basedir+'grid_pareto_costsU'+param_tag+'.npy',
                              basedir+'grid_pareto_costsC'+param_tag+'.npy']

    #param_tag = "_1.0_0.5_10000"
    #pareto_front_data_file = ['./Data/U_switch/plat/pareto_front'+param_tag+'_NSGAII_dtmax0.003.npy',
    #                          './Data/Cf_switch/plat/fixedGain/pareto_front'+param_tag+'_NSGAII_dtmax0.003.npy']

    #prob_dims = [(2,2),(2,2),(4,2)] # list of (Nvars,Nobjs) for each model
    prob_dims = [(2,2),(2,2)] # list of (Nvars,Nobjs) for each model

    Pareto_points = []
    Pareto_costs = []
    for i,pf in enumerate(pareto_front_data_file):
        pareto_data = np.load(pf)
        nv = prob_dims[i][0]
        Pareto_points.append(pareto_data[:,0:nv])
        Pareto_costs.append(pareto_data[:,nv:])

    plot_every = 5	# plot every n points on Pareto front

    #----------- Indices for phase-plane plotting (set manually) -------------#
    ind_sample = [-1040,-745]
    plot_inds = []
    for i,pareto_costs in enumerate(Pareto_costs):
        # indices of pareto-optimal costs sorted by C-cost values
        pcosts_Csorted_inds = sorted(range(len(pareto_costs)), key=lambda k: pareto_costs[k][0])

        # indices of the unsorted arrays corresponding to the sampled, sorted values
        plot_inds.append(pcosts_Csorted_inds[ind_sample[i]])

        print(pareto_costs[plot_inds[-1],:])
        print(Pareto_points[i][plot_inds[-1],:])
    print plot_inds
    
    #plot_inds = [48,46]
    #plot_inds = [44,54]
    #plot_inds = [44,63,20]

    #------------------------------------------------------------------------------------------------
    #----- Set up model for trajectory simulations -----#
    # Physical Parameters
    KCAT = 8.15e-4
    KM = 1.1e4
    VU = 200.
    VC = 60.
    B = 1.85e-4
    G = 5.0
    TAU_UPR = 15.*60.

    kcat = KCAT+B

    # Non-dimensional parameters
    nu = VU/VC
    alpha = kcat/B
    beta = KM*B/VC
    tau_upr = TAU_UPR*B

    # Pulse Parameters
    split_tag = param_tag.split('_')
    if split_tag[1]=='totaluin':
        d = float(split_tag[4])
        tau_p = float(split_tag[6])
    else:
        d = float(param_tag.split('_')[1]) #1. 
        tau_p = float(param_tag.split('_')[2]) #1000.0*60.
    teq = 30.*tau_upr

    umin = 1.0
    cmax = 1.0
    mu = 1.0
    mc = 1.0

    # Initialize dde solvers
    params_u = {
        'alpha' : alpha,
        'beta' : beta,
        'nu' : nu,
        'G' : G,
        'umin' : umin,
        'm' : mu,
        'tau_upr' : tau_upr,
        'd' : d,
        'tau_p' : tau_p,
        'teq' : teq
    }

    params_c = {
        'alpha' : alpha,
        'beta' : beta,
        'nu' : nu,
        'G' : G,
        'cmax' : cmax,
        'm' : mc,
        'tau_upr' : tau_upr,
        'd' : d,
        'tau_p' : tau_p,
        'teq' : teq
    }

    params = [params_u,params_c]

    #tfinal = 2000.*60 + teq
    tfinal = teq + tau_p + 30.*tau_upr

    dde_u = upw.initialize_dde_nf_upr(params=params_u,tfinal=tfinal)
    dde_c = cpw.initialize_dde_nf_upr(params=params_c,tfinal=tfinal) 

    #------------------------------------------------------------------------------------------------

    #----- Calculate Max U values along pareto front ------#
    umaxs = []
    DDEs = [dde_u,dde_c,dde_u]
    Params = [params_u,params_c,params_u]
    for i,(pareto_points,pareto_costs,dde,params) in enumerate(zip(Pareto_points,Pareto_costs,DDEs,Params)):

        #-----------------------------------------
        calc_umax = True
        if calc_umax==True:
            #sorted_points = pareto_points[pcosts_Csorted_inds]
            umax = []
            #for pi in sorted_points:
            for pi in pareto_points[0::plot_every,:]:
                params['m'] = pi[1]
                if 'umin' in params:
                    params['umin'] = pi[0]
                    soli = upw.run_dde(dde,params)
                if 'cmax' in params:
                    params['cmax'] = pi[0]
                    soli = cpw.run_dde(dde,params)
                umax.append(max(soli['u']))
        umaxs.append(umax)
        #-----------------------------------------

    umax_max = max([max(umaxi) for umaxi in umaxs])
    umax_min = min([min(umaxi) for umaxi in umaxs])

    blues = plt.get_cmap('Blues')

    #----- Plot pareto front -----#
    figP,ax = plt.subplots(1,1,figsize=(4,4))
    labels = [r'$U$-switch',r'$C_F$-switch',r'AND-Switch']
    colors = ['blue','red','green']
    markers = ['o','s','p']

    for i,(pareto_points,pareto_costs,umax) in enumerate(zip(Pareto_points,Pareto_costs,umaxs)):
        plot_ind = plot_inds[i]
        label = labels[i]
        color = colors[i]
        marker = markers[i]
        ms = 3

        # Mark specific points with green triangles
        for j in range(pareto_costs.shape[0]):
            if j == plot_ind:
                xpad = 0.03
                ypad = 0.05
                ax.plot(pareto_costs[j,0],pareto_costs[j,1],
                        linestyle='', marker='^', markeredgecolor='green',
                         markerfacecolor='green', markersize=ms+3, label=label)

                ax.annotate('', xy=(pareto_costs[j,0],pareto_costs[j,1]),
                        xytext=(pareto_costs[j,0]*(1+xpad),pareto_costs[j,1]*(1+ypad)),
                        arrowprops=dict(facecolor='black',lw=2,arrowstyle='->') )

        # plot pareto front as scatter plot w/ color based on Umax
        sc = ax.scatter(pareto_costs[0::plot_every,0],pareto_costs[0::plot_every,1],s=20,
                   c=umax,vmin=umax_min*0.8,vmax=umax_max,cmap=blues,
                   marker=marker,label=label)
                   
        # plot pareto front as scatter plot w/ marker size based on Umax
        #print(min(umax),max(umax))
        #sizemap = [5.*((max(umax)-umi)/max(umax) + 1.0)**8. for umi in umax]
        #print(min(sizemap),max(sizemap))
        #sc = ax.scatter(pareto_costs[:,0],pareto_costs[:,1],s=sizemap,
        #           marker=marker,label=label)


    cb = figP.colorbar(sc)
    tick_locator = mpl.ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.ax.set_title(r'$u_{max}$',fontsize=18)
    #cb.set_label(r'$u_{max}$',fontsize=18)

    ax.set_xlabel(r'$\mu_{C}$',fontsize=18)
    ax.set_ylabel(r'$\mu_{U}$',fontsize=18)

    ax.locator_params(axis='both', nbins=5)

    handles, labels = ax.get_legend_handles_labels()
    display_legend = (2,len(handles)-1)
    #display_legend = (0,int(len(handles)/2),len(handles)-1)
    ax.legend([handle for i,handle, in enumerate(handles) if i in display_legend],
                 [label for i,label in enumerate(labels) if i in display_legend])


    ax.set_xlim(1.0,1.3)
    #ax.set_ylim(1.12,ax.get_ylim()[1])

    figP.tight_layout()

    figP.savefig('./Figures/ParetoFrontCompare_U_Cf_wUmax'+param_tag+'.eps',edgecolor='black')
    #figP.savefig('./Figures/ParetoFrontCompare_U_Cf_AND'+param_tag+'.eps',edgecolor='black')

    plt.show()
