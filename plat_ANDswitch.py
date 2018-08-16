"""
This script calculates the Pareto front for the AND-switch mechanism using 
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
from delayed_nf_upr_piecewise_and_switch import *
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
    pulse_type = 'med'
    if pulse_type=='chronic':
        d = 1.0 
        tau_p = 1000.0*60.*B
    elif pulse_type=='acute':
        d = 100.0 
        tau_p = 10.0*60.*B
    elif pulse_type=='med':
        d = 1.0 
        tau_p = 50.0*60.*B

    totaluin = 0.5
    d = 1.0e+0
    tau_p = totaluin/d

    teq = 30.*tau_upr

    # Response function variables (values not used)
    mu = 1.0
    umin = 0.8
    mc = 1.0
    cmax = 0.6

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
    #tfinal = 2000.*60 + teq
    tfinal = teq + tau_p + 30.*tau_upr
    dtmax = 0.003
    dde = initialize_dde_nf_upr(params=params,tfinal=tfinal,dtmax=dtmax)
    USS, CSS = calc_steady_state(params)
    CF_SS = CSS*(1.-USS/(beta+USS))

    # Define objective function w/ constraints
    def andswitch_objective(vars):
        params['umin'] = vars[0]
        params['mu'] = vars[1]
        params['cmax'] = vars[2]
        params['mc'] = vars[3]
        obj = objective_function(dde,params)
        pad = 0.001
        u_constraint = USS*(1.+pad) - vars[0]
        cf_constraint = vars[2] - CF_SS*(1.+pad)
        return (obj,[u_constraint,cf_constraint])

    def alt_andswitch_objective(vars):
        params['umin'] = vars[0]
        params['mu'] = vars[1]
        params['cmax'] = vars[2]
        params['mc'] = vars[3]
        obj = objective_function(dde,params)
        pad = 0.01
        u_constraint = USS*(1.+pad) - vars[0]
        cf_constraint = vars[2] - CF_SS*(1.+pad)
        # if only one constraint is violoated, accept as feasible
        if (u_constraint>0. and cf_constraint<0.):
            u_constraint = -1.
        elif (cf_constraint>0. and u_constraint<0.):
            cf_constraint = -1.

        return (obj,[u_constraint,cf_constraint])

    # Define multi-objective optimization problem object
    #umin_min = USS*1.01
    umin_min = USS*0.5
    #umin_max = USS*1000.
    umin_max = USS*100.
    #cmax_min = CF_SS*1e-3
    cmax_min = CF_SS*1e-2
    #cmax_max = CF_SS*0.99	# for short pulse
    cmax_max = CF_SS*2.	# for short pulse
    mmin = 0.01
    #mmax = 1000.0
    mmax = 100.0
    problem = Problem(4,2,2)
    problem.types[:] = [Real(umin_min,umin_max),Real(mmin,mmax),
                        Real(cmax_min,cmax_max),Real(mmin,mmax)]
    problem.constraints[:] = "<=0"
    #problem.function = andswitch_objective
    problem.function = alt_andswitch_objective

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
            divs = 50
            alg_name=alg_name+str(divs)
            algorithm = NSGAIII(problem,divisions_outer=divs,evaluator=evaluator)
        elif alg_name=='SPEA2':
            algorithm = SPEA2(problem,evaluator=evaluator)
        else:
            print("Using NSGAII algorithm by default")
            algorithm = NSGAII(problem,evaluator=evaluator)
        algorithm.run(n_evals)
    print("Total run time = {:.2f} hours".format((time.time()-start_time)/3600.))

    # Save Results
    save_dir = "./Data/AND_switch/plat/"
    pareto_file_name = save_dir + "pareto_front_{0}_{1}_{2}_".format(d,tau_p,n_evals) + alg_name + "_dtmax{1}_relaxedBounds.npy".format(n_proc,dtmax)
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
