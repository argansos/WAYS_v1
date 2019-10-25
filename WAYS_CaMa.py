#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ganquan Mao (ganquan.mao@icloud.com)
# @Link    : https://argansos.github.io
# @Version : 3.6


import os
import glob
import time
import shutil
import subprocess
import numpy as np
import xarray as xr

from core import toolkit


# ==================== general configuration ====================
path_mod_simu = './data/simu/'
path_des_ = './data/CaMa/'

path_CaMa_Flood = './CaMa_Flood/'

path_CaMa_inp = os.path.join(path_CaMa_Flood, 'inp/WAYS/runoff_nc/')
file_syb = 'xx_ffff_hist_nosoc_co2_vvvv_global_daily_ssss_eeee.nc4'
variable = 'qtot'
forcing = 'gswp3'


# ==================== key function to run modules ====================
# ==================== key function to run modules ====================

def run(mod, time_s, time_e, rzsc_type='', pheix='', call='global_30min'):
    t1 = time.time()  # time it
    model = 'mod: ' + mod.lower()

    # time information
    time_ix = [time_s, time_e]

    year_start, year_end = toolkit.se_extract(time_ix[0], time_ix[1])
    ncfiles = toolkit.namestr_long(model, file_syb, forcing, variable, year_start, year_end)
    if mod.lower() == 'ways':
        path_simu = os.path.join(path_mod_simu, 'simu_' + rzsc_type + '_' + pheix)
    else:
        path_simu = os.path.join(path_mod_simu, 'ISIMIP2a', mod)
    files = toolkit.fullfile(path_simu, ncfiles['mod'])
    ds = xr.open_mfdataset(files)
    years = np.arange(int(time_ix[0][:4]), int(time_ix[1][:4]) + 1, 1)
    # prepare for data
    for year in years:
        ds1 = ds.sel(time=slice(str(year) + '-01-01', str(year) + '-12-31'))
        ds1 = ds1.rename({variable: 'runoff'})
        if mod.lower() != 'ways':
            if ds1.runoff.attrs['units'].lower() == 'kg m-2 s-1':
                prcp_ratio = 24 * 60 * 60
                ds1.runoff.attrs['units'] = 'mm'
            else:
                prcp_ratio = 1
            ds1.runoff.values *= prcp_ratio
        ds1.to_netcdf(os.path.join(path_CaMa_inp, 'runoff' + str(year) + '.nc'))

    t2 = time.time()  # time it
    print('Elapsed Time for data preparing:', (t2 - t1) / 3600, 'Hours')

    # run CaMa-Flood
    # os.chdir(os.path.join(path_CaMa_Flood, 'gosh'))
    subprocess.call([os.path.join(path_CaMa_Flood, 'gosh', './' + call + '.sh')])

    t3 = time.time()  # time it
    print('Elapsed Time for Calculating:', (t3 - t2) / 3600, 'Hours')

    # move the data to right folder
    # os.chdir(os.path.join('../out', call))

    if mod.lower() == 'ways':
        path_store = os.path.join(path_des_, 'discharge_' + rzsc_type + '_' + pheix + '_' + call)
    else:
        path_store = os.path.join(path_des_, mod.lower())

    if not os.path.exists(path_store):
        os.makedirs(path_store)

    for f in glob.glob(os.path.join('./CaMa_Flood/out/', call, '*.nc')):
        shutil.move(f, path_store)

    t4 = time.time()  # time it
    print('Elapsed Time for Calculating:', (t4 - t3) / 3600, 'Hours')


if __name__ == '__main__':
    run('H08', '1971-01-01', '2010-12-31', '', '', 'global_30min')
