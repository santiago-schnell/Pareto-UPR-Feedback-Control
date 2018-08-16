"""
This script generates plots comparing multiple phase planes and timecourses
for specified parameterization of the models. The data files (in .npy format) should be 
supplied as list of stings containing the path to the data files. Data files of the correct format
are generated from either the plat_* scripts or the gridsearch_* scripts that calculate Pareto fronts.
Additionally, the user should supply the indices of the points on the Pareto front for which the
phase planes and trajectories will be calculated.

Written by Wylie Stroberg in 2018
"""

#----------------------------------------------------------------------
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
mpl.rc('axes', linewidth=2)
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from pydelay import dde23
import delayed_nf_upr_piecewise_u_switch as upw
import delayed_nf_upr_piecewise_cf_switch as cpw
import plat_Uswitch as platU
import plat_Cfswitch as platC
from gridsearch_compare_switches import calc_costsU
#----------------------------------------------------------------------
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

    basedir = './Data/bruteforce/m100_uc100/totaluin_1.0/'
    param_tag = "_totaluin_1.0_d_2.15443469003_taup_0.464158883361"
    #param_tag = "_totaluin_1.0_d_0.1_taup_10.0"
    pareto_front_data_file = [basedir+'grid_pareto_costsU'+param_tag+'.npy',
                              basedir+'grid_pareto_costsC'+param_tag+'.npy']

    #param_tag = "_1.0_0.5_10000"
    #pareto_front_data_file = ['./Data/U_switch/plat/pareto_front'+param_tag+'_NSGAII_dtmax0.003.npy',
    #                          './Data/Cf_switch/plat/fixedGain/pareto_front'+param_tag+'_NSGAII_dtmax0.003.npy']

    Pareto_points = []
    Pareto_costs = []
    for pf in pareto_front_data_file:
        pareto_data = np.load(pf)
        Pareto_points.append(pareto_data[:,0:2])
        Pareto_costs.append(pareto_data[:,2:])

    # Set the indices in the Pareto_points array for which the analysis is performed
    plot_indices = [140, 772]

    free_params = [pp[plot_indices[i],0:2] for i,pp in enumerate(Pareto_points)]
    cost_vals = [pp[plot_indices[i],:] for i,pp in enumerate(Pareto_costs)]
    print cost_vals

    umin = free_params[0][0]
    mu = free_params[0][1]
    cmax = free_params[1][0]
    mc = free_params[1][1]

    print(free_params)
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

    params = [params_u,params_c]

    #tfinal = 2000.*60 + teq
    tfinal = teq + tau_p + 30.*tau_upr

    dde_u = upw.initialize_dde_nf_upr(params=params_u,tfinal=tfinal,dtmax=0.001)
    dde_c = cpw.initialize_dde_nf_upr(params=params_c,tfinal=tfinal,dtmax=0.001) 

    sols = [upw.run_dde(dde_u,params_u),cpw.run_dde(dde_c,params_c)]

    c_minmax = (min([np.min(soli['c']) for soli in sols]),max([np.max(soli['c']) for soli in sols]))
    u_minmax = (min([np.min(soli['u']) for soli in sols]),max([np.max(soli['u']) for soli in sols]))

    costs = [platU.objective_function(dde_u,params_u),platC.objective_function(dde_c,params_c)]

    USS,CSS = platU.calc_steady_state(params_u)
    CFSS = CSS*(1.-USS/(beta+USS))
    print(params_u)
    print("********************************************************")
    print("Maximal unfolded protein levels:")
    print("U-switch: {}".format(np.max(sols[0]['u'])))
    print("Cf-switch: {}".format(np.max(sols[1]['u'])))
    print("Steady-state concentrations:")
    print("Uss = {}, Css = {}, CFss = {}".format(USS,CSS,CFSS))
    print("********************************************************")

    # Calculate activation thresholds for each sensor model
    c_range = np.linspace(c_minmax[0],c_minmax[1],10)

    act_thresh_u = params_u['umin']*np.ones(c_range.shape)#act_mid_u
    act_thresh_c = params_c['beta']*(c_range/params_c['cmax'] - 1.)
    act_thresh = [act_thresh_u,act_thresh_c]

    #----- Plot switch function and dynamic response -----#
    fig,ax = plt.subplots(1,1,figsize=(4,4))
    fig_traj,ax_traj = plt.subplots(2,1,figsize=(5,4))

    colors = ['blue','red']
    labels = ['$U$-Switch','$C_{F}$-Switch']
    for i,soli in enumerate(sols):

        # Plot phase plane trajectories
        ax.plot(soli['c'],soli['u'],'-',color=colors[i],linewidth=2,label=labels[i])
        ax.plot(c_range,act_thresh[i],'--',color=colors[i])

        # Plot time courses
        start_ind = int(len(soli['t'])*0.1)
        end_ind = int(len(soli['t'])*0.9)
        shifted_time = soli['t'] - soli['t'][start_ind]
        ax_traj[0].plot(shifted_time[start_ind:end_ind],soli['u'][start_ind:end_ind],color=colors[i],label=labels[i])
        ax_traj[1].plot(shifted_time[start_ind:end_ind],soli['c'][start_ind:end_ind],color=colors[i],label=labels[i])


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
        ct = np.linspace(ax.get_xlim()[0],1.*ax.get_xlim()[1],40)
        u = np.linspace(ax.get_ylim()[0],1.*ax.get_ylim()[1],40)
        C,U = np.meshgrid(ct,u)
        positions = np.vstack([C.ravel(),U.ravel()])
        GU = []
        GC = []
        for i in range(positions.shape[1]):
            c = positions[0,i]
            u = positions[1,i]
            GU.append(g_u(u-params_u['umin'],params_u['m']))
            cf = c*(1.0 - u/(params_c['beta']+u))
            GC.append(g_c(cf-params_c['cmax'],params_c['m']))

        GU = np.reshape(np.array(GU),U.shape)
        GC = np.reshape(np.array(GC),U.shape)

        fig1,ax1 = plt.subplots(1,2,figsize=(8,4))
        zmin = 0.
        zmax = np.max([np.max(GU),np.max(GC)])

        cmap = plt.get_cmap('viridis')
        cmap = truncate_colormap(cmap,0.3,1.0)
        c0 = ax1[0].pcolormesh(C,U,GU,vmin=zmin,vmax=zmax,shading='gouraud',cmap=cmap)
        ax1[1].pcolormesh(C,U,GC,vmin=zmin,vmax=zmax,shading='gouraud',cmap=cmap)

        ax1[0].set_ylabel(r'$u$',fontsize=18)
        ax1[0].set_xlabel(r'$c_{T}$',fontsize=18)
        ax1[1].set_xlabel(r'$c_{T}$',fontsize=18)

        ax1[0].locator_params(axis='y',nbins=5)

        ax1[1].set_yticks([])

        for axi,soli in zip(ax1,sols):
            axi.plot(soli['c'],soli['u'],'-',color='black',linewidth=2)

        fig1.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.8,
                    wspace=0.1, hspace=0.02)
        cb_ax = fig1.add_axes([0.83, 0.15, 0.02, 0.75])
        cbar = fig1.colorbar(c0, cax=cb_ax)
        cbar.ax.set_title(r'$G/G_{0}$')

        fig1.savefig('./Figures/PhasePlane_wActLevel_U_Cf'+param_tag+'.pdf',edgecolor='black')
        #fig1.savefig('./Figures/PhasePlane_wActLevel_U_Cf_AND'+param_tag+'.pdf',edgecolor='black')


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

    ax_traj[0].legend()
    fig_traj.subplots_adjust(hspace=0.1)

    fig_traj.tight_layout()


    #fig.savefig('./Figures/PhasePlaneCompare_U_Cf'+param_tag+'.eps',edgecolor='black')
    #fig_traj.savefig('./Figures/TrajectoryCompare_U_Cf'+param_tag+'.eps',edgecolor='black')

    plt.show()
