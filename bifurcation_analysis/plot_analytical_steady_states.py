"""
Script that plots the analytical Pareto front in the long-stress (chronic) limit. 
Additionally, the activation functions for the U-switch and Cf-switch corresponding
to a point on the Pareto front are also plotted.

Written by Wylie Stroberg in 2018

"""

import numpy as np
import matplotlib.pyplot as plt
from plot_piecewise_switch import plot_switch
#-------------------------------------------------------------
def calc_unstressed_steady_state(params):
    nu = params['nu']
    alpha = params['alpha']
    beta = params['beta']

    css = 1.
    a1 = nu - alpha - beta
    uss = 0.5*(a1 + np.sqrt(a1**2. + 4*nu*beta))
    return uss,css
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
    Umin = 1e5
   
    kcat = KCAT+B
 
    # Nondimensional parameters
    alpha = kcat/B
    beta = KM*B/VC
    tau_upr = TAU_UPR*B
    G = G
    d = 2.
    nu0 = VU/VC
    nu = (1.+d)*nu0	# basal rate = Vu/Vc

    params = {'alpha':alpha,
              'beta':beta,
              'nu':nu}

    print("alpha = {}".format(alpha))
    print("beta = {}".format(beta))
    print("tau_upr = {}".format(tau_upr))

    # Calculate steady-state u concentration as function of steady-state c_T concentration
    def uStar(cstar,params):
        nu = params['nu']
        alpha = params['alpha']
        beta = params['beta']
        B = nu - alpha*cstar - beta
        return 0.5*(B + np.sqrt(B**2. + 4.*nu*beta))

    ustar_unstressed = uStar(1,{'nu':nu0,'alpha':alpha,'beta':beta})

    #cstar_max = nu0*d/alpha + 1.
    cstar_max = (nu-ustar_unstressed)*(beta + ustar_unstressed)/(alpha*ustar_unstressed)
    print("U_unstressed = {}".format(ustar_unstressed))
    print("Cstar_max = {}".format(cstar_max))

    cstar = np.linspace(1.,cstar_max,100)
    ustar = uStar(cstar,params)

    mu_u = ustar/ustar_unstressed - 1.
    mu_c = cstar - 1.

    ###### Plot subset of possible switching functions for specific point on pareto front ######
    ### Plot U-switch functions ###
    ind = int(len(ustar)/2)
    ui = ustar[ind]
    ci = cstar[ind]
    print(ui,ci)
    uminVals = [ustar_unstressed+vi*(ui-ustar_unstressed) for vi in [0.25, 0.75, 0.99]]
    mVals = [(ci-1.)/(G*(ui-umini)) for umini in uminVals]

    print("************************")
    print("U-switch Parameterizations:")
    print("Umin values: {}, {}, {}".format(*uminVals))
    print("m values: {}, {}, {}".format(*mVals))
    print("************************")

    figU, axU = plt.subplots(1,1)
    urange = np.linspace(0,uminVals[0]+1./mVals[0],1000)
    [plot_switch(umini,mi,x=urange,switch='uswitch',axis=axU) for (umini,mi) in zip(uminVals,mVals)]

    # Set ticks and labels for x-axis
    axU.set_xticks((urange[0],ui))
    label_text = ['' for i in axU.get_xticks()]
    label_text[0] = r"${:.1f}$".format( axU.get_xticks()[0])
    label_text[-1] = r"$u^{{*}}$".format( axU.get_xticks()[-1])
    axU.set_xticklabels(label_text)

    #### Plot Cf-switch functions ###
    def cf(u,c):
        return c*(1.-u/(beta+u)) # also equal to c*beta/(beta+u)

    cfi = cf(ui,ci)
    cf_unstressed = cf(ustar_unstressed,1.)
    maxCfmax = cfi*(G/(G-(ci-1.)))
    #cfmaxVals = [cf_unstressed - vi*(cf_unstressed-cfi) for vi in [0.99, 0.995, 0.999]]
    cfmaxVals = [maxCfmax - vi*(maxCfmax-cfi) for vi in [0.2, 0.6, 0.99]]
    #cfmaxVals = [1.1*cfi,1.05*cfi,1.01*cfi]
    mVals = [(ci-1.)/(G*(cfmaxi-cfi)) for cfmaxi in cfmaxVals]
    print(ci,cfi,cf_unstressed,maxCfmax)
    print("Cf-switch Parameterizations:")
    print("Cfmax values: {}, {}, {}".format(*cfmaxVals))
    print("m values: {}, {}, {}".format(*mVals))
    print("************************")

    figCf, axCf = plt.subplots(1,1)
    cfrange = np.linspace(0.0,cfmaxVals[0]*1.1,1000)
    [plot_switch(cfmaxi,mi,x=cfrange,switch='cfswitch',axis=axCf) for (cfmaxi,mi) in zip(cfmaxVals,mVals)]

    # Set ticks and labels for x-axis
    axCf.set_xticks((cfrange[0],cfi))
    label_text = ['' for i in axCf.get_xticks()]
    label_text[0] = r"${:.1f}$".format( axCf.get_xticks()[0])
    label_text[-1] = r"$c_{{F}}^{{*}}$".format( axCf.get_xticks()[-1])
    axCf.set_xticklabels(label_text)


    ######### Plot Pareto front ##########
    fig,ax = plt.subplots(1,1)

    ax.plot(mu_c,mu_u,'*')
    ax.plot(mu_c[ind],mu_u[ind],'^',markersize=10,markerfacecolor='red')

    ax.set_xlabel(r'$\mu_{c}$',fontsize=18)
    ax.set_ylabel(r'$\mu_{u}$',fontsize=18)

    ax.locator_params(axis='y',nbins=5)

    ax.tick_params(axis='both',labelsize=16)

    fig.tight_layout()
    figU.tight_layout() 
    figCf.tight_layout() 

    savefigures = True 
    if savefigures==True: 
        fig.savefig("./Figures/ChronicParetoFront_nu{:.2f}.eps".format(nu),edgecolor='black')
        figU.savefig("./Figures/ChronicUSwitches_nu{:.2f}.eps".format(nu),edgecolor='black')
        figCf.savefig("./Figures/ChronicCfSwitches_nu{:.2f}.eps".format(nu),edgecolor='black')
 
    plt.show()
