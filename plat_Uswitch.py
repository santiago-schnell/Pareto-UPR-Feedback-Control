"""
This script calculates the Pareto front for the U-switch mechanism using 
multi-objective optimzation methods provided in the package Platypus
(https://platypus.readthedocs.io/en/latest/).

Written by Wylie Stroberg in 2018
"""

import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
import time
from platypus import NSGAII, NSGAIII, SPEA2, Problem, Real, ProcessPoolEvaluator
from delayed_nf_upr_piecewise_u_switch import *
#------------------------------------------------------------
#------------------------------------------------------------

#------------------------------------------------------------
def objective_function(dde,params):

    sol = run_dde(dde,params)

    c = sol['c']
    u = sol['u']
    t = sol['t']

    USS, CSS = calc_steady_state(params)

    Cin = c[-1]-c[0] + simps(c,t)
    Udiff = u-USS
    Uexcess = np.where(Udiff<0,0,Udiff)
    Utot = simps(Uexcess,t)

    # Normalize
    cin = Cin/(t[-1])
    utot = Utot/(USS*t[-1]) + 1

    return [cin,utot]

#------------------------------------------------------------
#------------------------------------------------------------
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

    # Pulse Parameters
    pulse_type = 'short_acute'
    if pulse_type=='chronic':
        d = 1.0 
        tau_p = 1000.0*60.*B
    elif pulse_type=='acute':
        d = 100.0 
        tau_p = 10.0*60.*B
    elif pulse_type=='med':
        d = 1.0 
        tau_p = 50.0*60.*B
    elif pulse_type=='long_chronic':
        d = 1.0
        tau_p = 3000.0*60.*B
    elif pulse_type=='short_acute':
        d = 1000.0
        tau_p = 1.0*60.*B

    totaluin = 0.5
    d = 1.0e+0
    tau_p = totaluin/d

    teq = 30.*tau_upr

    # Response function variables (values not used)
    m = 1.0
    umin = 0.6

    # Initialize dde solver
    params = {
        'alpha' : alpha,
        'beta' : beta,
        'nu' : nu,
        'G' : G,
        'umin' : umin,
        'm' : m,
        'tau_upr' : tau_upr,
        'd' : d,
        'tau_p' : tau_p,
        'teq' : teq
    } 

    tfinal = teq + tau_p + 30.*tau_upr
    dtmax = 0.003

    dde = initialize_dde_nf_upr(params=params,tfinal=tfinal,dtmax=dtmax)
    USS, CSS = calc_steady_state(params)

    # Define objective function w/ constraints
    def uswitch_objective(vars):
        params['umin'] = vars[0]
        params['m'] = vars[1]
        obj = objective_function(dde,params)
        pad = 0.001
        constraint = USS*(1.+pad) - vars[0]
        return (obj,[constraint])


    # Define multi-objective optimization problem object
    umin_min = USS*1.01
    umin_max = USS*1000.	# for short pulse
    mmin = 0.01
    mmax = 1000.0
    problem = Problem(2,2,1)
    problem.types[:] = [Real(umin_min,umin_max),Real(mmin,mmax)]
    problem.constraints[:] = "<=0"
    problem.function = uswitch_objective

    # Run optimization in parallel
    n_proc = 4
    n_evals = 100000
    alg_name = 'NSGAII'
    #alg_name = 'NSGAIII'
    #alg_name = 'SPEA2'
    start_time = time.time()
    with ProcessPoolEvaluator(n_proc) as evaluator:
        if alg_name=='NSGAII':
            algorithm = NSGAII(problem,evaluator=evaluator)
        elif alg_name=='NSGAIII':
            divs = 20
            alg_name=alg_name+str(divs)
            algorithm = NSGAIII(problem,divisions_outer=div,evaluator=evaluator)
        elif alg_name=='SPEA2':
            algorithm = SPEA2(problem,evaluator=evaluator)
        else:
            print("Using NSGAII algorithm by default")
            algorithm = NSGAII(problem,evaluator=evaluator)
        algorithm.run(n_evals)
    print("Total run time = {:.2f} hours".format((time.time()-start_time)/3600.))

    # Save Results
    save_dir = "./Data/U_switch/plat/"
    pareto_file_name = save_dir + "pareto_front_{0}_{1}_{2}_".format(d,tau_p,n_evals) + alg_name + "_dtmax{0}.npy".format(dtmax)
    points = []
    costs = []
    for s in algorithm.result:
        points.append(s.variables[:])
        costs.append(s.objectives[:])
    points = np.array(points)
    costs = np.array(costs)
    pareto_front = np.hstack((points,costs))
    np.save(pareto_file_name,pareto_front)

    # Plot Pareto set
    plt.scatter([s.objectives[0] for s in algorithm.result],
                [s.objectives[1] for s in algorithm.result])
    plt.xlabel("$f_1(x)$")
    plt.ylabel("$f_2(x)$")

    plt.show()
