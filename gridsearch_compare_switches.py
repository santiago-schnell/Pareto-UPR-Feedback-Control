"""
Defines several funtions for running a brute-force determination
of the Pareto fronts of the U-switch and Cf-switch models.

Written by Wylie Stroberg in 2018

"""

import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
from platypus import NSGAII, Problem, Real, ProcessPoolEvaluator
from joblib import Parallel, delayed
import delayed_nf_upr_piecewise_u_switch as usw
import delayed_nf_upr_piecewise_cf_switch as csw
import delayed_nf_upr_piecewise_and_switch as asw
import plat_Uswitch as platU
import plat_Cfswitch as platC
import plat_ANDswitch as platA
from scipy.integrate import simps, quad, trapz
from scipy.interpolate import interp1d

#------------------------------------------------------------
#------------------------------------------------------------
def meshgrid2(*arrs):
    arrs = tuple(reversed(arrs))
    lens = map(len, arrs)
    dim = len(arrs)
    sz = 1
    for s in lens:
        sz *= s
    ans = []
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)
    return tuple(ans)

#----------------------------------------------------------------------
def is_pareto_efficient(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient

#----------------------------------------------------------------------
def integrate_pareto_fronts(pareto_costsU,pareto_costsC):
    """
    :param pareto_costsU, pareto_costs2: (n_points,n_costs) arrays of pareto sets
    : return: area under each curve, between smallest codomain shared between costs1 and costs2
              and truncated pareto_costs used for integration.
    """

    # Remove redundant points on pareto front
    empty, unique_paretoU = np.unique(pareto_costsU[:,0],return_index=True)
    empty, unique_paretoC = np.unique(pareto_costsC[:,0],return_index=True)

    unique_costsU = pareto_costsU[unique_paretoU,:]
    unique_costsC = pareto_costsC[unique_paretoC,:]

    # Create interpolation functions for each front
    fU = interp1d(unique_costsU[:,0],unique_costsU[:,1],kind='linear',fill_value='extrapolate')
    fC = interp1d(unique_costsC[:,0],unique_costsC[:,1],kind='linear',fill_value='extrapolate')

    # Find the codomain covered by both pareto fronts
    muC_lowerlim = np.max([np.min(unique_costsU[:,0]),np.min(unique_costsC[:,0])])
    muC_upperlim = np.min([np.max(unique_costsU[:,0]),np.max(unique_costsC[:,0])])

    # Remove points on pareto front outside of codomain
    unique_trunc_paretoU = np.where(np.logical_and(unique_costsU[:,0]>=muC_lowerlim, unique_costsU[:,0]<=muC_upperlim))
    unique_trunc_paretoC = np.where(np.logical_and(unique_costsC[:,0]>=muC_lowerlim, unique_costsC[:,0]<=muC_upperlim))

    costsU_trunc = unique_costsU[unique_trunc_paretoU,:][0]
    costsC_trunc = unique_costsC[unique_trunc_paretoC,:][0]

    # Interpolate any missing endpoints
    if muC_lowerlim not in costsU_trunc[:,0]:
        costsU_trunc = np.vstack((np.array([muC_upperlim,fU(muC_upperlim)]),costsU_trunc))

    if muC_lowerlim not in costsC_trunc[:,0]:
        costsC_trunc = np.vstack((np.array([muC_upperlim,fC(muC_upperlim)]),costsC_trunc))

    if muC_upperlim not in costsU_trunc[:,0]:
        costsU_trunc = np.vstack((costsU_trunc,np.array([muC_upperlim,fU(muC_upperlim)])))

    if muC_upperlim not in costsC_trunc[:,0]:
        costsC_trunc = np.vstack((costsC_trunc,np.array([muC_upperlim,fC(muC_upperlim)])))

    # Integrate along Pareto front
    IU = simps(costsU_trunc[:,1],costsU_trunc[:,0])
    IC = simps(costsC_trunc[:,1],costsC_trunc[:,0])

    return IU, IC, costsU_trunc, costsC_trunc

#----------------------------------------------------------------------
def calc_costsU(params,n_gridpoints=100):

    tfinal = params['teq'] + params['tau_p'] + 30.*params['tau_upr']

    dde_u = usw.initialize_dde_nf_upr(params=params,tfinal=tfinal,dtmax=0.001)
    USS, CSS = platU.calc_steady_state(params)

    # Define objective function
    def uswitch_objective(pdict):
        for key in pdict:
            params[key] = pdict[key]
        obj = platU.objective_function(dde_u,params)
        return obj

    # Define bounds for multi-objective optimization problem
    umin_min = USS*1.01
    umin_max = USS*1000.

    mmin = 1.0e-1
    mmax = 1.0e3

    urange = np.logspace(np.log10(umin_min),np.log10(umin_max),n_gridpoints)
    mrange = np.logspace(np.log10(mmin),np.log10(mmax),n_gridpoints)

    gU = meshgrid2(mrange,urange)
    pointsU = np.vstack(map(np.ravel, gU))
    resU = []
    Ukeys = ['umin','m']
    for point in zip(pointsU[0,:],pointsU[1,:]):
        pdict = dict(zip(Ukeys, point))
        resU.append(uswitch_objective(pdict))

    costsU = np.array(resU)


    return np.transpose(pointsU), costsU

#----------------------------------------------------------------------
def calc_costsC(params,n_gridpoints=100):

    tfinal = params['teq'] + params['tau_p'] + 30.*params['tau_upr']

    dde_c = csw.initialize_dde_nf_upr(params=params,tfinal=tfinal,dtmax=0.001)

    USS, CSS = platU.calc_steady_state(params)
    CFSS = CSS*(1.-USS/(params['beta']+USS))
    def cfswitch_objective(pdict):
        # check constraints
        constraint_cf = pdict['cmax']-CFSS # must be <0
        constraint_Gain = 1. - pdict['cmax']*pdict['m'] # must be <0
        if (constraint_cf < 0 and constraint_Gain < 0):
            for key in pdict:
                params[key] = pdict[key]
            obj = platC.objective_function(dde_c,params)
        else:
            obj = None
        return obj


    # Define bounds for multi-objective optimization problem
    cmax_min = CFSS*0.001
    cmax_max = CFSS*0.999

    mmin = 1.0e-1
    mmax = 1.0e3

    cfrange = np.logspace(np.log10(cmax_min),np.log10(cmax_max),n_gridpoints)
    mrange = np.logspace(np.log10(mmin),np.log10(mmax),n_gridpoints)

    gC = meshgrid2(mrange,cfrange)
    pointsC = np.vstack(map(np.ravel, gC))
    res_costs = []
    res_points = []
    Ckeys = ['cmax','m']
    for point in zip(pointsC[0,:],pointsC[1,:]):
        pdict = dict(zip(Ckeys,point))
        obj = cfswitch_objective(pdict)
        if obj is not None:
            res_costs.append(obj)
            res_points.append(point)

    return np.array(res_points), np.array(res_costs)

#----------------------------------------------------------------------
def calc_costsAND(params,n_gridpoints=100):

    tfinal = params['teq'] + params['tau_p'] + 30.*params['tau_upr']

    dde = asw.initialize_dde_nf_upr(params=params,tfinal=tfinal,dtmax=0.001)

    USS, CSS = platA.calc_steady_state(params)
    CFSS = CSS*(1.-USS/(params['beta']+USS))
    def andswitch_objective(pdict):
        # check constraints
        constraint_cf = pdict['cmax']-CFSS # must be <0
        constraint_Gain = 1. - pdict['cmax']*pdict['mc'] # must be <0
        if (constraint_cf < 0 and constraint_Gain < 0):
            for key in pdict:
                params[key] = pdict[key]
            obj = platA.objective_function(dde,params)
        else:
            obj = None
        return obj


    # Define bounds for multi-objective optimization problem
    umin_min = USS*1.01
    umin_max = USS*1000.

    cmax_min = CFSS*0.001
    cmax_max = CFSS*0.999

    mmin = 1.0e-1
    mmax = 1.0e3

    urange = np.logspace(np.log10(umin_min),np.log10(umin_max),n_gridpoints)
    murange = np.logspace(np.log10(mmin),np.log10(mmax),n_gridpoints)

    cfrange = np.logspace(np.log10(cmax_min),np.log10(cmax_max),n_gridpoints)
    mcrange = np.logspace(np.log10(mmin),np.log10(mmax),n_gridpoints)

    #g = meshgrid2(murange,urange,mcrange,cfrange)
    g = meshgrid2(mcrange,cfrange,murange,urange)
    points = np.vstack(map(np.ravel, g))
    res = []
    keys = ['umin','mu','cmax','mc']
    for point in zip(points[0,:],points[1,:],points[2,:],points[3,:]):
        pdict = dict(zip(keys,point))
        res.append(andswitch_objective(pdict))

    costs = np.array(res)

    return np.transpose(points), costs

#----------------------------------------------------------------------
#----------------------------------------------------------------------
def set_standard_params(pulse_type='acute'):

    # Physical Parameters
    KCAT = 8.15e-4
    KM = 1.1e4
    VU = 200.
    VC = 60.
    B = 1.85e-4
    G = 5.0
    TAU_UPR = 1.0e-0*15.*60.

    kcat = KCAT+B

    # Non-dimensional parameters
    nu = VU/VC
    alpha = kcat/B
    beta = KM*B/VC
    tau_upr = TAU_UPR*B

    m = 1.0
    umin = 0.6
    cmax = 2.0

    # Pulse Parameters
    if pulse_type=='chronic':
        d = 1.0 
        tau_p = 1000.0*60.*B
    elif pulse_type=='acute':
        d = 0.1 
        tau_p = 1000.*60.*B
    elif pulse_type=='med':
        d = 10.0 
        tau_p = 1.0*60.*B

    totaluin = 1.0
    d = 1.3
    tau_p = totaluin/d

    teq = 30.*tau_upr

    # Initialize dde solver
    params = {
        'alpha' : alpha,
        'beta' : beta,
        'nu' : nu,
        'G' : G,
        'umin' : umin,
        'cmax' : cmax,
        'm' : m,
        'tau_upr' : tau_upr,
        'd' : d,
        'tau_p' : tau_p,
        'teq' : teq
    } 

    return params
#------------------------------------------------------------
#------------------------------------------------------------
if __name__=="__main__":

    n_gridpoints = 20

    params = set_standard_params()

    print(platU.calc_steady_state(params))

    pointsU, costsU = calc_costsU(params,n_gridpoints)
    pointsC, costsC = calc_costsC(params,n_gridpoints)

    mmin = 1.0e-1
    mmax = 1.0e3
    mrange = np.logspace(np.log10(mmin),np.log10(mmax),n_gridpoints)
 
    paretoU = is_pareto_efficient(costsU)
    paretoC = is_pareto_efficient(costsC)

    pareto_costsU = costsU[paretoU]
    pareto_costsC = costsC[paretoC]

    ### Plot Pareto set ###
    fig, ax = plt.subplots(1,1)
    ax.scatter(costsU[:,0],costsU[:,1],s=np.array(range(len(mrange)))**1.,label='U-switch')
    ax.scatter(costsC[:,0],costsC[:,1],s=np.array(range(len(mrange)))**1.,label='Cf-switch')

    ax.scatter(costsU[paretoU,0],costsU[paretoU,1],s=np.array(range(len(mrange)))**1.,marker='s',color='black')
    ax.scatter(costsC[paretoC,0],costsC[paretoC,1],s=np.array(range(len(mrange)))**1.,marker='^',color='black')

    ax.set_xlabel("$f_1(x)$")
    ax.set_ylabel("$f_2(x)$")
    ax.legend()

    fig.tight_layout()

    #fig.savefig('Figures/gridsearch_ParetoFront_d_{}_taup_{}.eps'.format(d,tau_p),edgecolor='black')

    run_and_switch = False
    if run_and_switch:
        paramsAND = set_standard_params()
        paramsAND['mc'] = 0
        paramsAND['mu'] = 0
        
        pointsA, costsA = calc_costsAND(paramsAND,n_gridpoints)

        paretoA = is_pareto_efficient(costsA)
        pareto_costsA = costsA[paretoA]

        print(pointsA.shape)
        print(pointsA[0,:])
        print(costsA.shape)
        print(pareto_costsA.shape)

        for i,ci in enumerate(costsA[:,1]):
            if ci<1.001:
                print(costsA[i,:])
                print(pointsA[i,:])
        ax.scatter(costsA[:,0],costsA[:,1],s=np.array(range(len(mrange)))**1.,label='AND-switch')
        ax.scatter(costsA[paretoA,0],costsA[paretoA,1],s=np.array(range(len(mrange)))**1.,marker='.',color='black')

    ax.legend()

    fig.tight_layout()
        

    ####### Integrate area under Pareto fronts #######
    IU, IC, costsU_trunc, costsC_trunc = integrate_pareto_fronts(pareto_costsU,pareto_costsC)
    dI = IU-IC

    print(IU,IC)
    print("**************************************************")
    print("d = {}, tau_p = {}, IU - IC = {}".format(params['d'],params['tau_p'],dI))
    print("**************************************************")

    figP,axP = plt.subplots(1,1)

    axP.plot(costsU_trunc[:,0],costsU_trunc[:,1],'*-',linewidth=1)
    axP.plot(costsC_trunc[:,0],costsC_trunc[:,1],'*-',linewidth=1)

    axP.set_xlabel("$f_1(x)$")
    axP.set_ylabel("$f_2(x)$")


    plt.show()
