"""
This script calculates fixed points and plots bifucation diagrams in terms of the influx
rate, nu, for the Cf-switch model.

Written by Wylie Stroberg in 2018
"""

import numpy as np
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import delayed_nf_upr_piecewise_cf_switch as pw
from scipy.optimize import fsolve

#-------------------------------------------------------------
def calc_nu_crit_inactive(params):
    return (1.-params['cmax'])*(params['cmax']*params['alpha'] + params['beta'])/params['cmax']	# exact for discrete switch

#-------------------------------------------------------------
def calc_nu_crit_active(params):
    #return (1.+G-params['cmax'])*(params['cmax']*params['alpha'] + params['beta'])/params['cmax']	# exact for discrete switch
    return (1. + (1.+G-params['cmax'])*params['m']) * ((params['cmax']*params['alpha'] + params['beta'])*params['m'] - params['alpha'])/((params['cmax']*params['m']-1.)*params['m'])	# exact for discrete switch

#-------------------------------------------------------------
def calc_mean_and_oscillation_range(sol):
    Ueq = sol['u'][-20000:]
    Ceq = sol['c'][-20000:]
    Umean = np.mean(Ueq)
    Umin = np.min(Ueq)
    Umax = np.max(Ueq)
    Cmean = np.mean(Ceq)
    Cmin = np.min(Ceq)
    Cmax = np.max(Ceq)
    return( (Umin,Umax,Umean),(Cmin,Cmax,Cmean) )

#-------------------------------------------------------------
#-------------------------------------------------------------
if __name__=="__main__":

    # Physical Parameters
    KCAT = 8.15e-4
    KM = 1.1e4
    VU = 200
    VC = 60
    B = 1.85e-4
    G = 5.
    TAU_UPR = 1.e-0*15.*60.
   
    kcat = KCAT+B
 
    # Nondimensional parameters
    alpha = kcat/B
    beta = KM*B/VC
    tau_upr = TAU_UPR*B
    G = G

    #cmax = 0.1
    #m = 200.

    #cmax = 0.2# 0.2268 #0.250 #0.273
    #m = 10. #351.40 #8.785 #4.40

    #cmax = 0.46524
    #m = 2.57670
    #cmax = 0.425499
    #m = 5.1534
    cmax = 0.386747
    m = 206.13639

    print("alpha = {}".format(alpha))
    print("beta = {}".format(beta))
    print("cmax = {}".format(cmax))
    print("m = {}".format(m))
    print("tau_upr = {}".format(tau_upr))

    # bifurcation parameter
    n_nu_points = 40
    nu_high = 36.
    nu = np.linspace(1.,nu_high,n_nu_points) # VU/VC
    nu_fine = np.linspace(1.,nu_high,10*n_nu_points) # VU/VC

    # Initialize DDE solver
    params = {
        'alpha': alpha,
        'beta':  beta,
        'cmax':  cmax,
        'm':	m,
        'tau_upr': tau_upr,
        'G':     G,
        'nu':    nu[0]
    }    

    tfinal = 200	# end_time*B

    # Insert critical values into nu array for computaion
    nu_crit_inact = calc_nu_crit_inactive(params)
    nu_crit_act = calc_nu_crit_active(params)

    nu = np.insert(nu,np.searchsorted(nu,nu_crit_inact),nu_crit_inact)
    nu = np.insert(nu,np.searchsorted(nu,nu_crit_act),nu_crit_act)

    sols = []
    Ueq = np.zeros(nu.shape)
    Ceq = np.zeros(nu.shape)

    U_low_numeric = np.zeros(nu.shape)
    U_high_numeric = np.zeros(nu.shape)
    C_low_numeric = np.zeros(nu.shape)
    C_high_numeric = np.zeros(nu.shape)

    CSS = np.zeros(nu_fine.shape)
    USS = np.zeros(nu_fine.shape)
    CSS_int = np.zeros(nu_fine.shape)
    USS_int = np.zeros(nu_fine.shape)
    CSS_act = np.zeros(nu_fine.shape)
    USS_act = np.zeros(nu_fine.shape)

    for i,nui in enumerate(nu):
        params['nu'] = nui

        print("nu = {}, exponent C0 = {}".format(nui,-params["alpha"]/(nui-1.)))
        print("nu = {}, exponent C1 = {}".format(nui,params["alpha"]/(nui-1.-params["G"])))

        dde = pw.initialize_dde_nf_upr(params=params,tfinal=tfinal)
        sol = pw.run_dde(dde,params)

        sols.append(sol)

        osc_range = calc_mean_and_oscillation_range(sol)

        Ueq[i] = osc_range[0][2]
        Ceq[i] = osc_range[1][2]

        U_low_numeric[i] = osc_range[0][0]
        U_high_numeric[i] = osc_range[0][1]
        C_low_numeric[i] = osc_range[1][0]
        C_high_numeric[i] = osc_range[1][1]

    for i, nui in enumerate(nu_fine):
        params['nu'] = nui
        uss, css = pw.steady_state(params,0)
        uss_int, css_int = pw.steady_state(params,1)
        uss_act, css_act = pw.steady_state(params,2)
        USS[i] = uss
        CSS[i] = css
        USS_int[i] = uss_int
        CSS_int[i] = css_int
        USS_act[i] = uss_act
        CSS_act[i] = css_act

    print(USS_int.shape)
    print(nu_fine.shape)
    nu_crit_inact = calc_nu_crit_inactive(params)
    nu_crit_act = calc_nu_crit_active(params)
    print(nu_crit_inact,nu_crit_act)

    nu_USS_stable = np.where(nu_fine<=nu_crit_inact)
    nu_USS_unstable = np.where(nu_fine>nu_crit_inact)

    nu_USSint_stable = np.where(np.logical_and(nu_fine>=nu_crit_inact, nu_fine<=nu_crit_act))
    #nu_USSint_unstable = np.where(np.logical_not(np.logical_and(nu_fine>=nu_crit_inact, nu_fine<=nu_crit_act)))
    nu_USSint_unstable_low = np.where(nu_fine<=nu_crit_inact)
    nu_USSint_unstable_high = np.where(nu_fine>=nu_crit_act)

    nu_USSact_stable = np.where(nu_fine>nu_crit_act)
    nu_USSact_unstable = np.where(nu_fine<=nu_crit_act)

    #####################################
    #-------PLOTTING--------#

    figU,axU = plt.subplots(1,1,figsize=(4.5,4))
    figC,axC = plt.subplots(1,1,figsize=(4.5,4))

    # Plot unfolded protein steady states
    #axU.plot(nu,Ueq,'*')

    axU.plot(nu_fine[nu_USS_stable],USS[nu_USS_stable],color='black',linewidth=2)
    axU.plot(nu_fine[nu_USS_unstable],USS[nu_USS_unstable],'--',color='black',linewidth=1.5)

    axU.plot(nu_fine[nu_USSint_stable],USS_int[nu_USSint_stable],color='blue',linewidth=2)
    #axU.plot(nu_fine[nu_USSint_unstable_low],USS_int[nu_USSint_unstable_low],'--',color='green',linewidth=2)
    #axU.plot(nu_fine[nu_USSint_unstable_high],USS_int[nu_USSint_unstable_high],'--',color='green',linewidth=2)

    axU.plot(nu_fine[nu_USSact_stable],USS_act[nu_USSact_stable],color='black',linewidth=2)
    axU.plot(nu_fine[nu_USSact_unstable],USS_act[nu_USSact_unstable],'--',color='black',linewidth=1.5)

    # Plot chaperone steady states
    #axC.plot(nu,Ceq,'*')
    axC.plot(nu_fine[nu_USS_stable],CSS[nu_USS_stable],color='blue',linewidth=2)
    axC.plot(nu_fine[nu_USS_unstable],CSS[nu_USS_unstable],'--',color='blue',linewidth=2)

    axC.plot(nu_fine[nu_USSint_stable],CSS_int[nu_USSint_stable],color='green',linewidth=2)
    #axC.plot(nu_fine[nu_USSint_unstable_low],CSS_int[nu_USSint_unstable_low],'--',color='green',linewidth=2)
    #axC.plot(nu_fine[nu_USSint_unstable_high],CSS_int[nu_USSint_unstable_high],'--',color='green',linewidth=2)


    axC.plot(nu_fine[nu_USSact_stable],CSS_act[nu_USSact_stable],color='red',linewidth=2)
    axC.plot(nu_fine[nu_USSact_unstable],CSS_act[nu_USSact_unstable],'--',color='red',linewidth=2)

    # plot range of oscillations
    #axU.plot(nu,U_low_numeric,'--',color='black',linewidth=1)
    #axU.plot(nu,U_high_numeric,'--',color='black',linewidth=1)
    axC.plot(nu,C_low_numeric,'--',color='black',linewidth=1)
    axC.plot(nu,C_high_numeric,'--',color='black',linewidth=1)

    axU.fill_between(nu,U_low_numeric,U_high_numeric,color='blue',alpha=0.2,linestyle='-')

    #axU.set_ylim([0,np.max(Ueq)])
    axU.set_ylim([0,2.])


    axU.set_xlabel(r'$\nu$',fontsize=22)
    axU.set_ylabel(r'$u_{ss}$',fontsize=22)

    axC.set_xlabel(r'$\nu$',fontsize=22)
    axC.set_ylabel(r'$c_{ss}$',fontsize=22)


    axU.locator_params(axis='y',nbins=3)
    axC.locator_params(axis='y',nbins=4)

    axU.tick_params(axis='both',labelsize=18)
    axC.tick_params(axis='both',labelsize=18)

    figU.tight_layout()
    figC.tight_layout()

    figU.savefig('./Figures/bifurcationU_cf_cmax_{}_slope_{}_delay_{}.pdf'.format(cmax,m,tau_upr))


    print("beta = {}".format(beta))
    print("alpha = {}".format(alpha))
    print("cmax = {}".format(cmax))
    print("G = {}".format(G))
    print nu_crit_inact
    print nu_crit_act
    plt.show()
