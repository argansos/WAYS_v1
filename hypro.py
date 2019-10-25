"""
Hydrological process simulation
copyright: (c) 2017 by Ganquan Mao.
"""

import math
import numpy as np
import pandas as pd
from core import errlib


# arguments should be a vector rather than a matrix


def run(prcp, tas, pet, par, s_init=[0] * 5, simax_pkg=[]):
    """
    the core function of the WAYS model
    :param prcp: precipitation
    :param tas: temperature above surface
    :param pet: potential evaporation
    :param par: parameters
    :param s_init: initial state of the reservoirs
    :return: Ea, RZWS, R, Q, s_init
    """
    if any(np.isnan(par)) or any(np.isnan(prcp)) or any(np.isnan(tas)) or any(np.isnan(pet)):
        ea_l, su_l, r_l, q_l = ([np.NaN] * len(prcp) for i in range(4))
        s_init_e = s_init
        return ea_l, su_l, r_l, q_l, s_init_e

    # area reduction factor & storage capacity factor
    # de Jong and Jetten (2007) & van den Hoof et al. (2013)
    arf = 0.4
    csc = 0.2

    # extract the par
    tt, fdd, simax, ce, beta, d, tlagf, tlags, sftr, kff, kf, ks, crmax, rzsc = par

    if len(simax_pkg) > 0:
        from core import phenology
        simax_cal = True
        tmin, time_ix, lat, lu, igss = simax_pkg
        dateinfo = pd.date_range(start=time_ix[0], end=time_ix[1], freq='d')
    else:
        simax_cal = False

        # time_ix, lat, tmin, srz, rzsc, lu = simax_pkg
        # phenology.igs(dateinfo, tmin, srz, lat, rzsc, lu)

    si, sw, srz, sf, ss = s_init  # different reservoirs
    sw_l, si_l, srz_l, ea_l, rf_l, rs_l, q_l, igs_l = ([] for i in range(8))

    if simax_cal:
        if len(igss) > 0 and type(igss) != list:
            igs_l = igss.tolist()
        for p, t, e0, t_min, dinfo in zip(prcp, tas, pet, tmin, dateinfo):
            # loop
            pr, snowm, sw = prcp_sep(p, t, tt, fdd, sw)
            ptf, si = intercept(pr, si, simax)
            ei, si, pet_r = evap_i(e0, si, simax)
            infil, runoff = rainfall_parti(ptf, snowm, srz, rzsc, beta)
            eu, srz = evap_u(infil, pet_r, srz, rzsc, beta, ce)
            ea = ei + eu
            rf, rs = runoff_split(runoff, d)
            srz, ss = capillary(srz, ss, crmax, rzsc)
            igs = phenology.igs(dinfo, t_min, srz, lat, rzsc)
            srz_l.append(srz)
            ea_l.append(ea)
            rf_l.append(rf)
            rs_l.append(rs)
            igs_l.append(igs)
            # simax
            if len(igs_l) >= 21:
                igs_21 = sum(igs_l[-21:]) / len(igs_l[-21:])
                lai = phenology.lai(igs_21, lu)
                simax = arf * csc * lai
    else:
        for p, t, e0 in zip(prcp, tas, pet):
            # loop
            pr, snowm, sw = prcp_sep(p, t, tt, fdd, sw)

            ptf, si = intercept(pr, si, simax)
            ei, si, pet_r = evap_i(e0, si, simax)
            infil, runoff = rainfall_parti(ptf, snowm, srz, rzsc, beta)
            eu, srz = evap_u(infil, pet_r, srz, rzsc, beta, ce)
            ea = ei + eu
            rf, rs = runoff_split(runoff, d)
            srz, ss = capillary(srz, ss, crmax, rzsc)
            # sw_l.append(sw)
            # si_l.append(si)
            srz_l.append(srz)
            ea_l.append(ea)
            rf_l.append(rf)
            rs_l.append(rs)
        igs_l = []

    rfl_l, rsl_l = lagrunoff(rf_l, rs_l, tlagf, tlags)
    for rfl, rsl in zip(rfl_l, rsl_l):
        qff, qf, qs, sf, ss = runoff_routine(rfl, rsl, sf, ss, sftr, kff, kf, ks)
        q = qff + qf + qs
        q_l.append(q)
    r_l = [rf_x + rs_x for rf_x, rs_x in zip(rf_l, rs_l)]
    s_init_e = [si, sw, srz, sf, ss]

    return ea_l, srz_l, r_l, q_l, s_init_e, igs_l[-21:]


def runoff_obj(par_fit, prcp, tas, pet, ob_runoff, scale, par, warm_t=24, gof_ix=1, simax_pkg=[], time_ix=[]):
    """
    objective function for calibration on runoff
    timeix = ['1986-1-1', '1995-12-31']
    scale = 'D' / M' / 'Y'
    """
    for i in range(len(par_fit['ix'])):
        par[par_fit['ix'][i]] = par_fit['value'][i]

    _, _, r, _, _, _ = run(prcp, tas, pet, par, simax_pkg=simax_pkg)

    if scale == 'D':
        pass
    else:
        assert time_ix, 'time index should be provided in monthly or yearly scale!'

    # offset the time for model warming
    time_s = time_ix[0]
    time_e = time_ix[1]
    pdtime = pd.date_range(start=time_s, end=time_e)
    dt_runoff = pd.DataFrame(dict(runoff=r), pdtime)
    mt_runoff = dt_runoff.resample(scale).sum()

    # remove warm time data
    mt_runoff = mt_runoff[warm_t:]
    # if len(mt_runoff) != len(ob_runoff):
    #     raise ValueError('generated runoff time length should consider warmup time!')
    if gof_ix == 1:
        # 1 - NSE
        s = list(mt_runoff['runoff'])
        o = ob_runoff
        return errlib.nse1(s, o)
    elif gof_ix == 2:
        # RMSE
        return (sum([(x - y)**2 for x, y in zip(list(mt_runoff['runoff']), ob_runoff)]) / len(ob_runoff))**(1 / 2)
    elif gof_ix == 3:
        # sum of the ABS
        return abs(sum(list(mt_runoff['runoff'])) - sum(ob_runoff))


def qtot_obj(par_fit, prcp, tas, pet, ob_qtot, scale, par, warm_t=24, gof_ix=1, simax_pkg=[], time_ix=[]):
    """
    objective function for calibration on runoff
    timeix = ['1986-1-1', '1995-12-31']
    scale = 'D' / M' / 'Y'
    """
    for i in range(len(par_fit['ix'])):
        par[par_fit['ix'][i]] = par_fit['value'][i]

    _, _, _, q, _, _ = run(prcp, tas, pet, par, simax_pkg=simax_pkg)

    if scale == 'D':
        pass
    else:
        assert time_ix, 'time index should be provided in monthly or yearly scale!'

    # offset the time for model warming
    time_s = time_ix[0]
    time_e = time_ix[1]
    pdtime = pd.date_range(start=time_s, end=time_e)
    dt_qtot = pd.DataFrame(dict(qtot=q), pdtime)
    mt_qtot = dt_qtot.resample(scale).sum()

    # remove warm time data
    mt_qtot = mt_qtot[warm_t:]
    # if len(mt_runoff) != len(ob_runoff):
    #     raise ValueError('generated runoff time length should consider warmup time!')
    if gof_ix == 1:
        # 1 - NSE
        s = list(mt_qtot['qtot'])
        o = ob_qtot
        return errlib.nse1(s, o)
    elif gof_ix == 2:
        # RMSE
        return (sum([(x - y)**2 for x, y in zip(list(mt_qtot['qtot']), ob_qtot)]) / len(ob_qtot))**(1 / 2)
    elif gof_ix == 3:
        # sum of the ABS
        return abs(sum(list(mt_qtot['qtot'])) - sum(ob_qtot))


def prcp_sep(p, t, tt, fdd, sw):
    """rainfall snowfall seperation and snowmelt"""
    if t > tt:
        pr = p
        # ps = 0
        snowm = min(fdd * (t - tt), sw)
        sw -= snowm
    else:
        pr = 0
        ps = p
        snowm = 0
        sw += ps
    return pr, snowm, sw


def intercept(pr, si, simax=0):
    """interception"""
    # ptf: precipitation throughfall
    if simax == 0:
        si = 0
        ptf = pr
    else:
        if pr + si > simax:
            ptf = pr - (simax - si)
            si = simax
        else:
            ptf = 0
            si = si + pr
    return ptf, si


def evap_i(pet, si, simax):
    """evaporation from interception"""
    # set to 0 when FAO pet is negative (Wilcox & Sly (1976))
    if pet < 0:
        pet = 0
    if simax == 0:
        ei = 0
    else:
        ei = min(si, pet * (si / simax) ** (2 / 3))
    si = si - ei
    pet_r = pet - ei
    return ei, si, pet_r


def rainfall_parti(ptf, snowm, srz, rzsc, beta):
    """effective precipitation partition"""
    # pe: effective precipitation
    pe = ptf + snowm
    # runoff coefficient
    rcoeff = 1 - (1 - srz / (rzsc * (1 + beta))) ** beta
    # infiltration
    infil = pe * (1 - rcoeff)
    # water exceed rzsc
    if infil + srz > rzsc:
        infil = rzsc - srz
    # runoff
    runoff = pe - infil
    return infil, runoff


def evap_u(infil, pet_r, srz, rzsc, beta, ce):
    """evaporation from unsaturated soil"""
    srz += infil
    eu = pet_r * min(srz / (rzsc * (1 + beta) * ce), 1)
    eu = min(eu, srz)
    srz -= eu
    return eu, srz


def runoff_split(runoff, d):
    """split runoff into fast response and slow response"""
    rf = runoff * d
    rs = runoff - rf
    return rf, rs


def peaklag(lag):
    """lag between rainfall and peak flow"""
    c = range(1, math.ceil(lag) + 1)
    lagix = [x / sum(c) for x in c]
    return lagix


def lagrunoff(rf, rs, tlagf, tlags):
    """runoff dynamic under lag effect"""
    # rf rs need to be a list
    lagixf = peaklag(tlagf)
    lagixs = peaklag(tlags)
    rfl = np.convolve(lagixf, rf)
    rsl = np.convolve(lagixs, rs)
    return rfl[:len(rf)], rsl[:len(rs)]


def runoff_routine(rfl, rsl, sf, ss, sftr, kff, kf, ks):
    """runoff generation"""
    # qff: surface runoff
    # qf:  fast subsurface runoff
    # sf:  slow subsurface runoff
    sf += rfl
    qff = max(0, sf - sftr) / kff
    sf -= qff
    qf = sf / kf
    sf -= qf
    ss += rsl
    qs = ss / ks
    ss -= qs
    return qff, qf, qs, sf, ss


def capillary(srz, ss, crmax, rzsc):
    """capillary rise"""
    if rzsc - srz > crmax:
        srz += crmax
        ss -= crmax
    else:
        srz += rzsc - srz
        ss -= rzsc - srz
    return srz, ss


if __name__ == '__main__':
    prcp_ = [np.random.uniform(0, 10) for i in range(3652)]
    tas_ = [np.random.uniform(0, 10) for i in range(3652)]
    pet_ = [x * 1.2 for x in tas_]
    par_ = [0, 2, 0, 0.5, 0.3, 0.8, 1, 3, 100, 1.2, 3, 100, 0.5, 400]
    tmin = [x - 1 for x in tas_]
    time_ix = ['2001-01-01', '2010-12-31']
    lat = 60
    lu = 2
    igss = [1] * 21
    simax_pkg = [tmin, time_ix, lat, lu, igss]
    ea_l_, su_l_, r_l_, q_l_, s_init_e_, igs = run(prcp_, tas_, pet_, par_, simax_pkg=simax_pkg)
    # ea_l_, su_l_, r_l_, q_l_, s_init_e_, igs = run(prcp_, tas_, pet_, par_)
    # ob_runoff_ = [np.random.uniform(0, 10) for i in range(96)]
    # scale = 'M'
    # par_fit = {'ix': [5], 'value': [1.2]}
    # gof_v = runoff_obj(par_fit, prcp_, tas_, pet_, ob_runoff_, scale, par=par_, warm_t=24, gof_ix=1, simax_pkg=simax_pkg, time_ix=time_ix)
    # print(gof_v)
