"""
This script runs brute-force calculations of the Pareto fronts for a range of 
different stress input pulse shapes.

Written by Wylie Stroberg in 2018
"""

import numpy as np
from gridsearch_compare_switches import *
import time
import os
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
if __name__=="__main__":

    params = set_standard_params()

    total_uin = 1.
    #d = np.logspace(-1,1,10)
    #d = np.logspace(1.1,2,5)
    #d = [d[0]]
    d = [0.75,1.25,5.0]
    tau_p = np.array([total_uin/di for di in d])

    ngrid = 100 # n gridpoints in m and u/c directions (nxn total function evals)

    calc_U = True
    calc_C = False

    savedir = "./Data/bruteforce/m100_uc100/totaluin_{}".format(total_uin)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    tstart = time.time()

    for (di,tau_pi) in zip(d,tau_p):
        params['d'] = di
        params['tau_p'] = tau_pi

        try:
            if calc_U:
                pointsU, costsU = calc_costsU(params,ngrid)

                paretoU = is_pareto_efficient(costsU)

                points_costsU = np.hstack((pointsU,costsU))

                pareto_costsU = costsU[paretoU]
                pareto_pointsU = pointsU[paretoU]
                pareto_points_costsU = np.hstack((pareto_pointsU,pareto_costsU))

                np.save(savedir + "/grid_costsU_totaluin_{}_d_{}_taup_{}.npy".format(total_uin,di,tau_pi),points_costsU)
                np.save(savedir + "/grid_pareto_costsU_totaluin_{}_d_{}_taup_{}.npy".format(total_uin,di,tau_pi),pareto_points_costsU)

            if calc_C:
                pointsC, costsC = calc_costsC(params,ngrid)

                paretoC = is_pareto_efficient(costsC)

                points_costsC = np.hstack((pointsC,costsC))

                pareto_costsC = costsC[paretoC]
                pareto_pointsC = pointsC[paretoC]

                pareto_points_costsC = np.hstack((pareto_pointsC,pareto_costsC))

                np.save(savedir + "/grid_costsC_totaluin_{}_d_{}_taup_{}.npy".format(total_uin,di,tau_pi),points_costsC)
                np.save(savedir + "/grid_pareto_costsC_totaluin_{}_d_{}_taup_{}.npy".format(total_uin,di,tau_pi),pareto_points_costsC)

            print("***************************************************")
            print("Completed cost calculation for d = {}, tau_p = {}".format(di,tau_pi))
            print("Total elapsed time: {}".format(time.time()-tstart))

        except Exception as e: 
            print("***************************************************")
            print("Failed on cost calculation for d = {}, tau_p = {}".format(di,tau_pi))
            print("Total elapsed time: {}".format(time.time()-tstart))
            print(e)

    print("***************************************************")
    
