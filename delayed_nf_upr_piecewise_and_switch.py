"""
Defines several funtions, including the model equations, that facilitate initializing and
running the delay differential equation model for the AND-switch activation mechanism.

Written by Wylie Stroberg in 2018

"""

import numpy as np
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
from pydelay import dde23
from joblib import Parallel, delayed

#------------------------------------------------------------
def calc_steady_state(params):
    nu = params['nu']
    alpha = params['alpha']
    beta = params['beta']

    css = 1.
    a1 = nu - alpha - beta
    uss = 0.5*(a1 + np.sqrt(a1**2. + 4*nu*beta))
    return uss,css

#--------------------------------------------------------------
def initialize_dde_nf_upr(**kwargs):
    # parse input
    params = kwargs['params']
    tfinal = kwargs['tfinal']
    dtmax = 0.01 
    if 'dtmax' in kwargs:
        dtmax = kwargs['dtmax']

    # define equations
    eqns = {
        'u' : '-alpha*c*u/(beta+u) + nu*(1. + d*(heaviside(t-teq) - heaviside(t-teq-tau_p))) - u',
        'c' : '1. + G*g_u(u(t-tau_upr)-umin,mu)*g_c(cf(c(t-tau_upr),u(t-tau_upr),beta)-cmax,mc) - c' }

    # Helper c functions
    mycode = """
    double heaviside(double t) {
        if(t>=0)
            return 1.0;
        else
            return 0.0;
    }

    double g_u(double x, double m) {
        if(x<=0.0)
            return 0.0;
        else if(x>=1.0/m)
            return 1.0;
        else
            return x*m;
    }

    double cf(double c, double u, double K) {
        return c*(1.0 - u/(K+u));
    }

    double g_c(double x, double m) {
        if(x>=0.0)
            return 0.0;
        else if(x<=-1.0/m)
            return 1.0;
        else
            return -x*m;
    }
    """

    # Initialize Solver
    dde = dde23(eqns=eqns, params=params, supportcode=mycode)

    # Set simulation parameters
    # (solve from t=0 to t=tfinal and limit the maximum step size to 1.0)
    dde.set_sim_params(tfinal=tfinal, dtmax=dtmax)

    # Set History function
    set_history(dde,params)

    return dde

#-------------------------------------------------------------
def set_history(dde,params):
    ''' Set history for dde to be the steady state of the unperturbed model.'''
    # Set History function
    #alpha = params['alpha']
    #beta = params['beta']
    #nu = params['nu']
    #umin = params['umin']
    #m = params['m']
    #G = params['G']

    uss, css = calc_steady_state(params)

    #if uss<0:
    #    print (uss,umin,m)
    #    uss = css

    histfunc = {
        'u' : lambda t: uss,
        'c' : lambda t: css
    }

    dde.hist_from_funcs(histfunc, 51)
    return

#-------------------------------------------------------------
def run_dde(dde,params,reset_hist=False):
    # set dde parameters
    for key in params:
        dde.params[key] = params[key]
   
    # set dde history function
    if reset_hist==True:
        set_history(dde,params)
 
    # Run Simulation
    dde.run()

    # Plot
    tstep = 0.001
    sol = dde.sample(None,None,tstep)

    return sol

#--------------------------------------------------------------
#--------------------------------------------------------------
if __name__=="__main__":

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

    mu = 9.71289066e+01 #1.0e3
    umin = 3.16713296e+00 #1.242e-2

    mc = 1.46349624e+02 #1.0e2
    cmax = 2.24717393e-02 #5.23e1

    # Pulse Parameters
    totaluin = 1.0
    d = 1.0e2 
    #tau_p = 1000.0*60.*B
    tau_p = totaluin/d
    teq = 30.*tau_upr

    # Initialize dde solver
    params = {
        'alpha' : alpha,
        'beta' : beta,
        'nu' : nu,
        'G' : G,
        'umin' : umin,
        'mu' : mu,
        'cmax' : cmax,
        'mc' : mc,
        'tau_upr' : tau_upr,
        'd' : d,
        'tau_p' : tau_p,
        'teq' : teq
    }

    tfinal = teq + tau_p + 30.*tau_upr
    dde = initialize_dde_nf_upr(params=params,tfinal=tfinal)
    uss, css = calc_steady_state(params)

    print(uss,css)

    sol = run_dde(dde,params)

    fig,ax = plt.subplots(2,1,figsize=(6,6))

    ax[0].plot(sol['t'],sol['u'])
    ax[1].plot(sol['t'],sol['c'])

    ax[0].plot(sol['t'],uss*np.ones(sol['u'].shape),'--')
    ax[1].plot(sol['t'],css*np.ones(sol['c'].shape),'--')

    plt.show()
