#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ganquan Mao (ganquan.mao@icloud.com)
# @Link    : https://argansos.github.io
# @Version : 3.6


import os
import glob
import time
import numpy as np
import pandas as pd
import xarray as xr

from core import toolkit
from core import errlib

# ==================== general configuration ====================
# path
path_simu = './data/simu/'
path_fluxnet = './data/FLUXNET/FLUXNET_FULLSITE_DD'
file_fluxnetsite = './data/FLUXNET/FLUXNET_SITE.csv'

# setting
model = 'ways: ways'
forcing = 'gswp3'
variable = 'evap'
file_syb = 'xx_ffff_hist_nosoc_co2_vvvv_global_daily_ssss_eeee.nc4'


# ==================== fluxcobine module ====================
# ==================== fluxcobine module ====================

def fluxcombine(site_id):
    fv = flux[site_id]
    if not fv.empty:
        lat1 = site.loc[site_id]['LOCATION_LAT']
        lon1 = site.loc[site_id]['LOCATION_LONG']
        [i, j] = toolkit.get_coor(lat1, lon1)
        d = var[:, i, j]
        fv = fv.join(pd.DataFrame({'WAYS': d}, index=pd.date_range(start=time_ix[0], end=time_ix[1], freq='D')))
        if all(np.isnan(d)):
            print(lat1, lon1, ' no data!')
        # montly mean
        if montly_do:
            fv = fv.resample('M').apply(np.nanmean)
            m_days = [x.to_pydatetime().day for x in fv.index]
            fv = fv * np.array(m_days)[:, None]

    # return values
    return site_id, fv


# ==================== key function to run modules ====================
# ==================== key function to run modules ====================

def run(time_s, time_e, rzsc_type, pheix, scale='M'):
    from multiprocessing import Pool

    global montly_do, var, site, flux, time_ix

    if scale == 'M':
        montly_do = True

    path_evap = os.path.join(path_simu, 'simu_' + rzsc_type + '_' + pheix)

    # time information
    time_ix = [time_s, time_e]

    # prepare the file name for reading
    year_start, year_end = toolkit.se_extract(time_ix[0], time_ix[1])
    ncfiles = toolkit.namestr_long(model, file_syb, forcing, variable, year_start, year_end)

    files = toolkit.fullfile(path_evap, ncfiles['ways'])

    # FLUXNET
    # site information
    files_fluxnet = glob.glob(os.path.join(path_fluxnet, '*.csv'))
    site = pd.read_csv(file_fluxnetsite)
    site = site.sort_values('SITE_ID')[['SITE_ID', 'LOCATION_LAT', 'LOCATION_LONG']]
    site = site.set_index('SITE_ID')

    # extract flux values & change the unit to mm/day
    flux = dict()
    for file in files_fluxnet:
        key = os.path.basename(file).rsplit('_')[1]
        value = toolkit.read_fluxnet(file, time_ix)
        flux[key] = value

    # evap simulation
    ds = xr.open_mfdataset(files)
    ds = ds.sel(time=slice(time_ix[0], time_ix[1]))
    var = ds.evap.values
    # caution: assume start from the first of the year and end in the last day
    anave_evap = np.sum(var, axis=0) / (int(time_e[:4]) - int(time_s[:4]) + 1)

    # sites
    sites = site.index.tolist()
    # processing
    t1 = time.time()  # time it
    pool = Pool()
    results = pool.map(fluxcombine, sites)
    pool.close()
    t2 = time.time()  # time it
    print('Elapsed Time for Calculation:', (t2 - t1) / 3600, 'Hours')

    # data in pandas dataframe (FLUXNET & WAYS)
    data = dict()
    for element in results:
        data[element[0]] = element[1]

    # statistics
    ops = site.copy()
    header = ['N', 'SLOP', 'INTERCEPT', 'MEAN_OBS', 'MEAN_SIM', 'R', 'P', 'STD_ERR', 'RMSE', 'NRMSE1']
    ops = ops.reindex(columns=ops.columns.tolist() + header)
    for site_id in site.index:
        try:
            fv = data[site_id]
        except KeyError:
            print(site_id + ': no observation is found!')
        fv = fv.dropna()
        N = len(fv)
        if fv.empty:
            slope, intercept, r_value, p_value, std_err, rmse, nrmse = [np.NaN] * 7
        else:
            o = fv['LE_CORR'].tolist()
            s = fv['WAYS'].tolist()
            slope, intercept, r_value, p_value, std_err = errlib.linergress(s, o)
            m_o = np.nanmean(o)
            m_s = np.nanmean(s)
            rmse, nrmse = errlib.rmse(s, o)
        ops.loc[site_id] = [ops.loc[site_id].LOCATION_LAT, ops.loc[site_id].LOCATION_LONG] + [N, slope, intercept, m_o, m_s, r_value, p_value, std_err, rmse, nrmse]

    # return values
    return anave_evap, data, ops
