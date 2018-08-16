"""
This script generates plots comparing multiple phase planes and timecourses
for specified parameterization of the U-switch, Cf-switch and AND-switch models. The data files
(in .npy format) should be supplied as list of stings containing the path to the data files. 
Data files of the correct format are generated from either the plat_* scripts or the gridsearch_* 
scripts that calculate Pareto fronts. Additionally, the user should supply the indices of the
points on the Pareto front for which the phase planes and trajectories will be calculated.

Written by Wylie Stroberg in 2018
"""

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
mpl.rc('axes', linewidth=2)
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from pydelay import dde23
import delayed_nf_upr_piecewise_u_switch as upw
import delayed_nf_upr_piecewise_cf_switch as cpw
import delayed_nf_upr_piecewise_and_switch as apw
import plat_Uswitch as platU
import plat_Cfswitch as platC
import plat_ANDswitch as platA
from gridsearch_compare_switches import calc_costsU
#----------------------------------------------------------------------
def g_u(x,m):
    ''' Input: x - u-umin
               m - slope
        Returns: normalized response between 0 and 1.
    '''
    if x <= 0.0:
        return 0.0
    elif x >= 1.0/m:
        return 1.0
    else:
        return x*m
#----------------------------------------------------------------------
def g_c(x,m):
    ''' Input: x - cf-cmax
               m - slope
        Returns: normalized response between 0 and 1.
    '''
    if x>=0.0:
        return 0.0
    elif x <= -1.0/m:
        return 1.0
    else:
        return -x*m

#----------------------------------------------------------------------
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mplcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

#----------------------------------------------------------------------
#----------------------------------------------------------------------
if __name__=="__main__":

    # Specifiy paths for input data files
    param_tag = "_1.0_0.5_100000"
    pareto_front_data_file = ['./Data/U_switch/plat/pareto_front'+param_tag+'_NSGAII_dtmax0.003.npy',
                              './Data/Cf_switch/plat/fixedGain/pareto_front'+param_tag+'_NSGAII_dtmax0.003.npy',
                              './Data/AND_switch/plat/fixedGain/pareto_front'+param_tag+'_NSGAII_dtmax0.003_relaxedBounds.npy']

    prob_dims = [(2,2),(2,2),(4,2)] # list of (Nvars,Nobjs) for each model

    Pareto_points = []
    Pareto_costs = []
    for i,pf in enumerate(pareto_front_data_file):
        pareto_data = np.load(pf)
        nv = prob_dims[i][0]
        Pareto_points.append(pareto_data[:,0:nv])
        Pareto_costs.append(pareto_data[:,nv:])
        print("Parameters at minimum mu_U:")
        print(pareto_data[np.argmin(pareto_data[:,nv+1]),:])

    plot_indices = [73,77,26]		# Points for plot 1 (fig 4b)
    #plot_indices = [76,23,36]		# Points for plot 2 (fig 4c)
    free_params = [pp[plot_indices[i],:] for i,pp in enumerate(Pareto_points)]
    cost_vals = [pp[plot_indices[i],:] for i,pp in enumerate(Pareto_costs)]

    umin = free_params[0][0]
    mu = free_params[0][1]
    cmax = free_params[1][0]
    mc = free_params[1][1]
    uminAnd = free_params[2][0]
    muAnd = free_params[2][1]
    cmaxAnd = free_params[2][2]
    mcAnd = free_params[2][3]

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

    params_a = {
        'alpha' : alpha,
        'beta' : beta,
        'nu' : nu,
        'G' : G,
        'umin' : uminAnd,
        'mu' : muAnd,
        'cmax' : cmaxAnd,
        'mc' : mcAnd,
        'tau_upr' : tau_upr,
        'd' : d,
        'tau_p' : tau_p,
        'teq' : teq
    }

    params = [params_u,params_c,params_a]

    #tfinal = 2000.*60 + teq
    tfinal = teq + tau_p + 30.*tau_upr

    dde_u = upw.initialize_dde_nf_upr(params=params_u,tfinal=tfinal,dtmax=0.001)
    dde_c = cpw.initialize_dde_nf_upr(params=params_c,tfinal=tfinal,dtmax=0.001) 
    dde_a = apw.initialize_dde_nf_upr(params=params_a,tfinal=tfinal,dtmax=0.001) 

    sols = [upw.run_dde(dde_u,params_u),cpw.run_dde(dde_c,params_c),apw.run_dde(dde_a,params_a)]

    c_minmax = (min([np.min(soli['c']) for soli in sols]),max([np.max(soli['c']) for soli in sols]))
    u_minmax = (min([np.min(soli['u']) for soli in sols]),max([np.max(soli['u']) for soli in sols]))

    costs = [platU.objective_function(dde_u,params_u),platC.objective_function(dde_c,params_c),platA.objective_function(dde_a,params_a)]
    print(costs)

    USS,CSS = platU.calc_steady_state(params_u)
    CFSS = CSS*(1.-USS/(beta+USS))
    print("********************************************************")
    print("Maximal unfolded protein levels:")
    print("U-switch: {}".format(np.max(sols[0]['u'])))
    print("Cf-switch: {}".format(np.max(sols[1]['u'])))
    print("Steady-state concentrations:")
    print("Uss = {}, Css = {}, CFss = {}".format(USS,CSS,CFSS))
    print("********************************************************")

    #------ Calculate threshold values for activation --------#
    c_range = np.linspace(c_minmax[0],c_minmax[1],10)

    act_thresh_u = params_u['umin']*np.ones(c_range.shape)
    act_thresh_c = params_c['beta']*(c_range/params_c['cmax'] - 1.)

    act_thresh_uAND =  params_a['umin']*np.ones(c_range.shape)
    act_thresh_cAND = params_a['beta']*(c_range/params_a['cmax'] - 1.)

    act_thresh = [act_thresh_u,act_thresh_c,[act_thresh_uAND,act_thresh_cAND]]

    #----- Plot switch function and dynamic response -----#
    fig,ax = plt.subplots(1,1,figsize=(4,4))
    fig_traj,ax_traj = plt.subplots(2,1,figsize=(5,4))

    colors = ['blue','red','green']
    labels = ['$U$-Switch','$C_{F}$-Switch','AND-Switch']
    plot_thresholds = [True,True,True]
    for i,soli in enumerate(sols):

        ax.plot(soli['c'],soli['u'],'-',color=colors[i],linewidth=2,label=labels[i])
        if plot_thresholds[i]:
            if isinstance(act_thresh[i],(list,)):
                [ax.plot(c_range,threshj,'--',color=colors[i]) for threshj in act_thresh[i]]
            else:
                ax.plot(c_range,act_thresh[i],'--',color=colors[i])
        start_ind = int(len(soli['t'])*0.1)
        end_ind = int(len(soli['t'])*0.9)
        shifted_time = soli['t'] - soli['t'][start_ind]
        ax_traj[0].plot(shifted_time[start_ind:end_ind],soli['u'][start_ind:end_ind],color=colors[i])
        ax_traj[1].plot(shifted_time[start_ind:end_ind],soli['c'][start_ind:end_ind],color=colors[i])

    # Set axis properties for phase plane
    ax.set_xlabel(r'$c_{T}$',fontsize=18)
    ax.set_ylabel(r'$u$',fontsize=18)

    ax.legend()

    ax.locator_params(axis='both', nbins=5)

    pad = 0.05
    ax.set_xlim((c_minmax[0]*(1.-pad),c_minmax[1]*(1.+pad)))
    ax.set_ylim((u_minmax[0]*(1.-pad),u_minmax[1]*(1.+pad)))

    fig.tight_layout()

    # Plot activation levels as heat map on phase-plane trajectories axis
    plot_activation_levels = True
    if plot_activation_levels:
        ct = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],40)
        u = np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],40)
        C,U = np.meshgrid(ct,u)
        positions = np.vstack([C.ravel(),U.ravel()])
        GU = []
        GC = []
        GA = []
        for i in range(positions.shape[1]):
            c = positions[0,i]
            u = positions[1,i]
            GU.append(g_u(u-params_u['umin'],params_u['m']))
            cf = c*(1.0 - u/(params_c['beta']+u))
            GC.append(g_c(cf-params_c['cmax'],params_c['m']))
            GA.append(g_u(u-params_a['umin'],params_a['mu'])*g_c(cf-params_a['cmax'],params_a['mc']))

        GU = np.reshape(np.array(GU),U.shape)
        GC = np.reshape(np.array(GC),U.shape)
        GA = np.reshape(np.array(GA),U.shape)

        fig1,ax1 = plt.subplots(1,3,figsize=(10,4))
        zmin = 0.
        zmax = np.max([np.max(GU),np.max(GC),np.max(GA)])

        cmap = plt.get_cmap('viridis')
        cmap = truncate_colormap(cmap,0.3,1.0)

        c0 = ax1[0].pcolormesh(C,U,GU,vmin=zmin,vmax=zmax,shading='gouraud',cmap=cmap)
        ax1[1].pcolormesh(C,U,GC,vmin=zmin,vmax=zmax,shading='gouraud',cmap=cmap)
        ax1[2].pcolormesh(C,U,GA,vmin=zmin,vmax=zmax,shading='gouraud',cmap=cmap)

        ax1[0].set_ylabel(r'$u$',fontsize=18)
        ax1[0].set_xlabel(r'$c_{T}$',fontsize=18)
        ax1[1].set_xlabel(r'$c_{T}$',fontsize=18)
        ax1[2].set_xlabel(r'$c_{T}$',fontsize=18)

        ax1[1].set_yticks([])
        ax1[2].set_yticks([])


        for axi,soli in zip(ax1,sols):
            axi.plot(soli['c'],soli['u'],'-',color='black',linewidth=2)

        fig1.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.8,
                    wspace=0.1, hspace=0.02)
        cb_ax = fig1.add_axes([0.83, 0.15, 0.02, 0.75])
        cbar = fig1.colorbar(c0, cax=cb_ax)
        cbar.ax.set_title(r'$G/G_{0}$')

        fig1.savefig('./Figures/PhasePlane_wActLevel_U_Cf_AND_1'+param_tag+'.pdf',edgecolor='black')
        #fig1.savefig('./Figures/PhasePlane_wActLevel_U_Cf_AND_2'+param_tag+'.pdf',edgecolor='black')
        

    # Set axis properties for trajectories
    ax_traj[1].set_xlabel(r'Time',fontsize=18)
    ax_traj[0].set_ylabel(r'$u$',fontsize=18)
    ax_traj[1].set_ylabel(r'$c_{T}$',fontsize=18)

    ax_traj[0].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    ax_traj[0].locator_params(axis='both',nbins=5)
    ax_traj[1].locator_params(axis='both',nbins=5)

    fig_traj.subplots_adjust(hspace=0.1)

    fig_traj.tight_layout()


    #fig.savefig('./Figures/PhasePlaneCompare_U_Cf_AND_2'+param_tag+'.eps',edgecolor='black')
    #fig_traj.savefig('./Figures/TrajectoryCompare_U_Cf_AND_2'+param_tag+'.eps',edgecolor='black')

    plt.show()
