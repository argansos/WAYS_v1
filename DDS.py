#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ganquan Mao (ganquan.mao@icloud.com)
# @Link    : https://argansos.github.io
# @Version : 3.6


import numpy as np
import math as m
from core import toolkit  #


def serial(forcing, obj_setups, gof_setups, par_setups, print_out=False):
    # setups parsing

    if any(np.isnan(forcing['obs'])) or any(np.isnan(forcing['par'])):
        return np.full(par_setups.size, np.NaN, dtype=np.double)

    fit_ix = gof_setups['fit_ix']
    mm_ix = gof_setups['mm_ix']
    its = gof_setups['its']
    maxiter = gof_setups['maxiter']
    sinitial = gof_setups['sinitial']

    # number of parameters
    nop = par_setups.size
    # candidate solution array
    stest = np.empty(nop, dtype=float)
    # best solution array
    sbest = np.empty_like(stest)
    # array of par_setups types(flt/int)
    dflag = par_setups['dflag']
    # array of par_setups ranges
    S_range = par_setups['S_max'] - par_setups['S_min']
    # number of iterations
    ileft = maxiter - its
    # its = number of function evaluations to initialize the DDS algorithm
    solution = np.empty((maxiter, nop + 3), dtype=float)

    # Initialization
    # modified on the code from Thouheed A.G.
    for i in range(its):
        if its > 1:
            if not dflag.all():  # handling continuous variables
                # return continuous uniform random samples
                stest = par_setups['S_min'] + S_range * np.random.random(nop)
            else:  # handling discrete case
                for j in range(nop):
                    # return random integers from the discrete uniform dist'n
                    stest[j] = np.random.randit([par_setups['S_min'][j], par_setups['S_max'][j]], size=1)
        else:  # know its=1, using a user supplied initial solution.
            # get initial solution from the input file
            stest = sinitial

        # Call obj function
        par_fit = {'ix': fit_ix, 'value': stest}
        Jtest = mm_ix * get_obj(forcing, obj_setups, par_fit)
        # Update current best
        if i == 0:
            Jbest = Jtest
        if Jtest <= Jbest:
            Jbest = Jtest
            np.copyto(sbest, stest)

        # Store initial sol. data in Master solution array
        solution[i, 0] = i
        solution[i, 1] = mm_ix * Jbest
        solution[i, 2] = mm_ix * Jtest
        solution[i, 3:3 + nop] = stest

    # Main loop
    for i in range(ileft):
        # probability of being selected as neighbour
        Pn = 1.0 - m.log1p(i) / m.log(ileft)
        # counter for how many decision variables vary in neighbour
        dvn_count = 0
        # define stest initially as current (sbest for greedy)
        np.copyto(stest, sbest)
        # Generate array of random uniformly distributed numbers for neighborhood inclusion
        randnums = np.random.random(nop)

        for j in range(nop):
            # then j th par_setups selected to vary in neighbour
            if randnums[j] < Pn:
                dvn_count = dvn_count + 1
                stest[j] = toolkit.perturb(sbest[j], par_setups['S_min'][j], par_setups['S_max'][j], par_setups['dflag'][j])

        # no DVs selected at random, so select ONE
        if dvn_count == 0:
            # which dec var to modify for neighbour
            dec_var = int(m.floor((nop) * np.random.random(1)))
            stest[dec_var] = toolkit.perturb(sbest[dec_var], par_setups['S_min'][dec_var], par_setups['S_max'][dec_var], par_setups['dflag'][dec_var])

        # Get ojective function value
        par_fit = {'ix': fit_ix, 'value': stest}
        Jtest = mm_ix * get_obj(forcing, obj_setups, par_fit)

        # Update current best
        if Jtest <= Jbest:
            Jbest = Jtest
            np.copyto(sbest, stest)
            # iteration number best solution found
            # it_sbest = i + its

        # accumulate results in Master output matrix
        # [col 0: iter # col 1: Fbest col 2: Ftest col 3: param set (xtest)]
        solution[i + its, 0] = i + its
        solution[i + its, 1] = mm_ix * Jbest
        solution[i + its, 2] = mm_ix * Jtest
        solution[i + its, 3:3 + nop] = stest

        if print_out:
            print(solution[i + its, :])

        if i + its >= 100:
            if np.mean(solution[i + its - 30: i + its, 1]) == 0:
                break
            if (abs(solution[i + its, 1] - solution[i + its - 30, 1]) * 100) / np.mean(solution[i + its - 30: i + its, 1]) < 0.0001:
                break

    return sbest


def get_obj(forcing, obj_setups, par_fit):
    '''
    get the result of objective funtion with the provided data and parameter
    objf_name: 'runoff'/'evaporation'
    '''
    import hypro

    prcp = forcing['prcp']
    tas = forcing['tas']
    pet = forcing['pet']
    obs = forcing['obs']
    par = forcing['par']
    simax_pkg = forcing['simax_pkg']
    time_ix = forcing['time_ix']
    objf_name = obj_setups['objf_name']
    gof_ix = obj_setups['gof_ix']
    scale = obj_setups['scale']
    warm_t = obj_setups['warm_t']

    if objf_name == 'runoff':
        feval = getattr(hypro, 'runoff_obj')
        return feval(par_fit, prcp, tas, pet, obs, scale, par, warm_t, gof_ix, simax_pkg, time_ix)
    elif objf_name == 'qtot':
        feval = getattr(hypro, 'qtot_obj')
        return feval(par_fit, prcp, tas, pet, obs, scale, par, warm_t, gof_ix, simax_pkg, time_ix)


if __name__ == '__main__':
    prcp_ = [np.random.uniform(0, 10) for i in range(3652)]
    tas_ = [np.random.uniform(0, 10) for i in range(3652)]
    pet_ = [x * 1.2 for x in tas_]
    ob_runoff_ = [np.random.uniform(0, 10) for i in range(96)]
    par_ = [0, 2, 0, 0.5, 0.3, 0.8, 1, 3, 100, 1.2, 3, 100, 400]
    tmin = [x - 1 for x in tas_]
    time_ix = ['2001-01-01', '2010-12-31']
    lat = 60
    lu = 2
    igss = [1] * 21
    simax_pkg = [tmin, time_ix, lat, lu, igss]
    forcing = {'prcp': prcp_, 'tas': tas_, 'pet': pet_, 'obs': ob_runoff_, 'par': par_, 'simax_pkg': simax_pkg, 'time_ix': time_ix}
    # par_fit = {'ix': [3, 4], 'value': [1, 2]}
    # get_obj(forcing, obj_setups, par_fit)
    obj_setups = toolkit.read_obj_setups('./configs/obj_setups.txt')
    gof_setups = toolkit.read_gof_setups('./configs/gof_setups.txt')
    par_setups = toolkit.read_par_setups('./configs/par_setups.txt')
    gof_r = serial(forcing, obj_setups, gof_setups, par_setups, print_out=True)
    print(gof_r)
