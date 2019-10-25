#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ganquan Mao (ganquan.mao@icloud.com)
# @Link    : https://argansos.github.io
# @Version : 3.6

'''
some critical functions
'''


import os
import re
import math
import glob
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import convolve1d
import statsmodels.formula.api as smf

import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict
from mpl_toolkits.axes_grid1 import make_axes_locatable

from core import errlib


def glob_re(pattern, strings):
    '''
    filenames = glob_re(r'.*(abc|123|a1b).*\.txt', os.listdir())
    '''
    return list(filter(re.compile(pattern).match, strings))


def se_extract(time_start, time_end, step=10):
    assert (step in [1, 10]), 'step must be 1 or 10'
    # split start year and end year in a 1-year/10-year datasets
    year_start = str(math.floor(int(time_start[:4]) / step) * step + 1)
    year_end = str(math.ceil(int(time_end[:4]) / step) * step)

    # to avoid issue that the start year large than end year
    if int(year_start) > int(time_start[:4]):
        year_start = str(int(year_start) - step)

    return year_start, year_end


def se_fill(time_start, time_end, step=10):
    assert (step in [1, 10]), 'step must be 1 or 10'
    [year_start, year_end] = se_extract(time_start, time_end)
    n = int((int(year_end) - int(year_start) + 1) / step)

    if n > 1:
        year_start = list(range(int(year_start), int(year_end) + 1, step))
        year_end = [(x + step - 1) for x in year_start]
    else:
        year_start = [int(year_start)]
        year_end = [int(year_end)]

    return year_start, year_end


def namestr(var_id, file_syb, forcing, year_start, year_end, step=10):
    assert (step in [1, 10]), 'step must be 1 or 10'
    n = int((int(year_end) - int(year_start) + 1) / step)

    if n > 1:
        year_start = list(range(int(year_start), int(year_end) + 1, step))
        year_end = [(x + step - 1) for x in year_start]
    else:
        year_start = [int(year_start)]
        year_end = [int(year_end)]

    ncfiles = []
    var_id = re.split(',\s|,|:\s|:', var_id)
    for i in var_id[1::2]:
        f_syb = []
        for j in range(n):
            file_syb1 = file_syb
            file_syb1 = file_syb1.replace('xx', i)
            file_syb1 = file_syb1.replace('ffff', forcing)
            file_syb1 = file_syb1.replace('ssss', str(year_start[j]))
            file_syb1 = file_syb1.replace('eeee', str(year_end[j]))
            f_syb.append(file_syb1)
        ncfiles.append(f_syb)
    ncfiles = dict(zip(var_id[0::2], ncfiles))

    return ncfiles


def namestr_long(model, file_syb, forcing, variable, year_start, year_end):
    n = int((int(year_end) - int(year_start) + 1) / 10)

    if n > 1:
        year_start = list(range(int(year_start), int(year_end), 10))
        year_end = [(x + 9) for x in year_start]
    else:
        year_start = [int(year_start)]
        year_end = [int(year_end)]

    ncfiles = []
    model_id = re.split(',\s|,|:\s|:', model)
    for i in model_id[1::2]:
        f_syb = []
        for j in range(n):
            file_syb1 = file_syb
            file_syb1 = file_syb1.replace('xx', i)
            file_syb1 = file_syb1.replace('ffff', forcing)
            file_syb1 = file_syb1.replace('vvvv', variable)
            file_syb1 = file_syb1.replace('ssss', str(year_start[j]))
            file_syb1 = file_syb1.replace('eeee', str(year_end[j]))
            f_syb.append(file_syb1)
        ncfiles.append(f_syb)
    ncfiles = dict(zip(model_id[0::2], ncfiles))

    return ncfiles


def fullfile(data_dir, ncfiles):
    # fullfile the file path
    fullpathnc = []
    if type(ncfiles) != list:
        ncfiles = [ncfiles]
    for ncfile in ncfiles:
        fullpathnc.append(os.path.join(data_dir, ncfile))

    return fullpathnc


def slice_timeix(pd_tstamp, t1, t2):
    # generate time index for data slicing
    # pd_tstamp: pandas time stamp of the complete ts
    # t1: pandas time stamp of the start time
    # t2: pandas time stamp of the end time
    pd_df = pd.DataFrame(index=pd_tstamp)
    pd_df['ix'] = range(len(pd_tstamp))
    time_ix = pd_df.ix[t1:t2]

    return [i for i in time_ix['ix']]


def ascii(file, start_line=6):
    # read ascii file, defaul hearline number is 6
    # return numpy 2-d array [col, row]
    from astropy.io import ascii
    dat = ascii.read(file, data_start=start_line)

    return np.array([dat[c] for c in dat.columns])


def read_obj_setups(file):
    A = np.loadtxt(file, dtype='S40', comments='#', skiprows=2)
    A = A.astype(str)

    return {'objf_name': A[0], 'gof_ix': int(A[1]), 'scale': A[2], 'warm_t': int(A[3])}


def read_gof_setups(file):
    A = np.loadtxt(file, dtype='S10', comments='#', skiprows=2)
    A = A.astype(str)

    return {'fit_ix': [int(x.strip()) for x in A[0].split(',')], 'mm_ix': int(A[1]), 'its': int(A[2]), 'maxiter': int(A[3]), 'rseed': int(A[4]), 'sinitial': [float(x.strip()) for x in A[5].split(',')]}


def read_par_setups(file):

    return np.loadtxt(file, dtype={'names': ('S_name', 'S_min', 'S_max', 'dflag'), 'formats': ('S3', 'f4', 'f4', 'i4')}, skiprows=1, ndmin=1)


def perturb(s, s_min, s_max, dflag):
    # modified on the code from Thouheed A.G.
    # Define parameter range
    s_range = s_max - s_min
    # Scalar neighbourhood size perturbation parameter (r)
    r = 0.2
    # Perturb variable
    z_value = stand_norm()
    delta = s_range * r * z_value
    s_new = s + delta

    # Handle perturbations outside of decision variable range:
    # Reflect and absorb decision variable at bounds

    # probability of absorbing or reflecting at boundary
    P_Abs_or_Ref = np.random.random()

    if dflag == 0:
        # Case 1) New variable is below lower bound
        if s_new < s_min:  # works for any pos or neg s_min
            if P_Abs_or_Ref <= 0.5:  # with 50% chance reflect
                s_new = s_min + (s_min - s_new)
            else:  # with 50% chance absorb
                s_new = np.copy(s_min)
            # if reflection goes past s_max then value should be s_min since without reflection
            # the approach goes way past lower bound.  This keeps X close to lower bound when X current
            # is close to lower bound:
            if s_new > s_max:
                s_new = np.copy(s_min)

        # Case 2) New variable is above upper bound
        elif s_new > s_max:  # works for any pos or neg s_max
            if P_Abs_or_Ref <= 0.5:  # with 50% chance reflect
                s_new = s_max - (s_new - s_max)
            else:  # with 50% chance absorb
                s_new = np.copy(s_max)
            # if reflection goes past s_min then value should be s_max for same reasons as above
            if s_new < s_min:
                s_new = np.copy(s_max)

    else:
        # Case 1) New variable is below lower bound
        if s_new < s_min - 0.5:  # works for any pos or neg s_min
            if P_Abs_or_Ref <= 0.5:  # with 50% chance reflect
                s_new = (s_min - 0.5) + ((s_min - 0.5) - s_new)
            else:  # with 50% chance absorb
                s_new = np.copy(s_min)
            # if reflection goes past s_max+0.5 then value should be s_min since without reflection
            # the approach goes way past lower bound.  This keeps X close to lower bound when X current
            # is close to lower bound:
            if s_new > s_max + 0.5:
                s_new = np.copy(s_min)

        # Case 2) New variable is above upper bound
        elif s_new > s_max + 0.5:  # works for any pos or neg s_max
            if P_Abs_or_Ref <= 0.5:  # with 50% chance reflect
                s_new = (s_max + 0.5) - (s_new - (s_max + 0.5))
            else:  # with 50% chance absorb
                s_new = np.copy(s_max)
            # if reflection goes past s_min -0.5 then value should be s_max for same reasons as above
            if s_new < s_min - 0.5:
                s_new = np.copy(s_max)
        # Round new value to nearest integer
        s_new = np.around(s_new)
        # Handle case where new value is the same as current: sample from
        # uniform distribution
        if s_new == s:
            samp = s_min - 1 + np.ceil(s_range) * np.random.rand()
            if samp < s:
                s_new = samp
            else:
                s_new = samp + 1

    return s_new


def stand_norm():
    # modified on the code from Thouheed A.G.
    # generate a standard Gaussian random number (zvalue)
    # based upon Numerical recipes gasdev and Marsagalia-Bray Algorithm
    Work3 = 2.0
    while((Work3 >= 1.0) or (Work3 == 0.0)):
        # call random_number(ranval) # get one uniform random number
        ranval = np.random.random()  # harvest(ign)
        Work1 = 2.0 * ranval - 1.0  # .0 * DBLE(ranval) - 1.0
    # call random_number(ranval) # get one uniform random number
        ranval = np.random.random()  # harvest(ign+1)
        Work2 = 2.0 * ranval - 1.0  # 2.0 * DBLE(ranval) - 1.0
        Work3 = Work1 * Work1 + Work2 * Work2
    Work3 = ((-2.0 * math.log(Work3)) / Work3)**0.5  # natural log

    # pick one of two deviates at random (don't worry about trying to use both):
    # call random_number(ranval) # get one uniform random number
    ranval = np.random.random()  # harvest(ign)
    if (ranval < 0.5):
        zvalue = Work1 * Work3
    else:
        zvalue = Work2 * Work3

    return zvalue


def ll2ij(coord):
    # lat, lat
    lon = np.linspace(-179.75, 179.75, 720)
    lat = np.linspace(89.75, -89.75, 360)
    i = lat.tolist().index(coord[0])
    j = lon.tolist().index(coord[1])
    return [i, j]


def ij2ll(ij):
    lon = np.linspace(-179.75, 179.75, 720)
    lat = np.linspace(89.75, -89.75, 360)
    lx = lat[ij[0]]
    ly = lon[ij[1]]
    return [lx, ly]


def read_fluxnet(file, time_ix=[]):
    df = pd.read_csv(file)[['TIMESTAMP', 'LE_CORR']].set_index('TIMESTAMP')
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    if len(time_ix) > 0:
        time_start = pd.to_datetime(time_ix[0])
        time_end = pd.to_datetime(time_ix[1])
        df = slice_df(df, [time_start, time_end])
    df = df.replace(-9999, np.nan)
    return df * 0.03527


def slice_df(df, dateix):
    if df.index[0] <= dateix[0]:
        start_date = dateix[0]
    else:
        start_date = df.index[0]
    if df.index[-1] >= dateix[-1]:
        end_date = dateix[-1]
    else:
        end_date = df.index[-1]
    return df.loc[start_date:end_date]


def get_coor(lat_input, lon_input, method='nearest'):
    # works on 0.5 degree spatial resolution
    lon = np.linspace(-179.75, 179.75, 720)
    lat = np.linspace(89.75, -89.75, 360)

    lat_index = np.nanargmin((lat - lat_input) ** 2)
    lon_index = np.nanargmin((lon - lon_input) ** 2)

    # correct to the xarray dataset loc function
    # if lat_input in lat + 0.25:
    #     lat_index += 1
    if lon_input in lon + 0.25:
        lon_index += 1

    return lat_index, lon_index


def upscale(x, ratio, method='mean', skipna=False):
    # x should be numpy array
    t = x.reshape(x.shape[0] // ratio, ratio, x.shape[1] // ratio, ratio)

    if method == 'mean':
        if not skipna:
            st = np.nanmean(t, axis=(1, 3))
        else:
            st = np.mean(t, axis=(1, 3))
    elif method == 'sum':
        st = np.sum(t, axis=(1, 3))

    return st


def downscal(x, ratio):
    return np.kron(x, np.ones((ratio, ratio)))


def read_GRDC_dir(path_src, header=37, time_ix=[]):
    files = glob.glob(os.path.join(path_src, '*.txt'))
    DATA = {}
    for file in files:
        ops = read_GRDC_main(file, header, time_ix=time_ix)
        DATA[ops['GRDC_No']] = ops

    return DATA


def read_GRDC_main(file, header, time_ix=[]):
    ops = read_GRDC_meta(file, header)
    ops['data'] = read_GRDC_data(file, header, time_ix=time_ix)

    return ops


def read_GRDC_data(file, header, time_ix=[]):
    df = pd.read_csv(file, header=header, delimiter=';').set_index('YYYY-MM-DD')
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    if len(time_ix) > 0:
        time_start = pd.to_datetime(time_ix[0])
        time_end = pd.to_datetime(time_ix[1])
        df = slice_df(df, [time_start, time_end])
    # update the index name
    df.index.names = ['TIMESTAMP']
    # update the colume names
    keys = df.keys()
    # keys for replacement (remove space)
    keys_new = [key.strip() for key in keys]
    # change index for keys
    chix = dict(zip(keys, keys_new))
    # updated df
    df = df.rename(columns=chix)
    # add a new column based on original value
    df['Value'] = df['Original']
    # replace the data with calculated one that used data frequency larger than 50%
    df.loc[(df['Calculated'] != -999) & (df['Flag'] >= 50), 'Value'] = df['Calculated']
    # fill_value -999 -> NaN
    df = df.replace(-999, np.nan)

    return df


def read_GRDC_meta(file, lastl):
    meta = {}
    namelist = ('# GRDC-No', '# River', '# Station', '# Latitude', '# Longitude', '# Catchment area')
    with open(file, 'rb') as f:
        for i, line in enumerate(f):
            li = line.decode("utf-8", errors='ignore')
            if li.startswith(namelist) and ':' in li:
                key, value = li.split(':', 1)
                name = key.strip('# .')
                name = name.replace('(DD)', '')
                name = name.replace('(km)', '')
                name = name.replace('-', '_')
                name = name.replace(' ', '_')
                meta[name.strip('_#')] = value.strip()
            if i > lastl:
                break
    f.close()
    return meta


def grdc663(grdc663_csv):
    # read meta information from grdc663 file that can be extracted via Qgis
    return pd.read_csv('./auxiliary/grdc663.csv')[['GRDC-ID', 'X-COORD', 'Y-COORD', 'RECORDNAME', 'STNINTERSTATIONA', 'INTERSTNA']].set_index('GRDC-ID')


def pos_corr(data_cama, data_up, id, time, k=1, move=[], crit='R', showfig=False):
    # p_VAL_RUNOFF.ipynb
    # k - search range

    # processing
    ij = get_coor(float(data_up[id]['Latitude']), float(data_up[id]['Longitude']))
    ll = ij2ll(ij)

    # manu correct
    if len(move) > 0:
        ij_new = [x + y for x, y in zip(ij, move)]
        criteria = []

    else:
        # select the correct position
        criteria = []
        for istep in np.arange(-k, k + 1).tolist():
            for jstep in np.arange(-k, k + 1).tolist():
                i = ij[0] + istep
                j = ij[1] + jstep
                # grdc
                df = data_up[id].copy()
                df_sub = df['data']
                df_sub = df_sub.drop(columns=['CaMa'])
                if len(df_sub) == 0 or all(np.isnan(df_sub['Value'])):
                    continue
                try:
                    df['data'].index = df['data'].index.strftime('%Y-%m')
                except AttributeError:
                    pass
                # cama
                var = data_cama[:, i, j]
                df_sub_cama = pd.DataFrame({'CaMa': var}, index=time)
                df_sub_cama.index = df_sub_cama.index.strftime('%Y-%m')
                df['data'] = df_sub.join(df_sub_cama)
                var_o = df['data']['Value']
                var_s = df['data']['CaMa']
                # R
                _, _, R, _, _ = errlib.linergress(var_s, var_o)
                # NSE
                NSE = 1 - errlib.nse1(var_s, var_o, obj=False)
                # annual average difference
                diff = abs(np.nanmean(var_s) - np.nanmean(var_o)) * 12
                row = [i, j, R, NSE, diff]
                criteria.append(row)

        if crit == 'R':
            crit_n = 2
        elif crit == 'NSE':
            crit_n = 3
        elif crit == 'DIFF':
            crit_n = 4
        criteria = np.array(criteria)
        if crit == 'DIFF':
            result = np.where(criteria == np.amin(criteria[:, crit_n]))
        else:
            result = np.where(criteria == np.amax(criteria[:, crit_n]))
        ij_new = [int(criteria[result[0], 0]), int(criteria[result[0], 1])]
        criteria = criteria.tolist()

    df = data_up[id].copy()
    var = data_cama[:, ij_new[0], ij_new[1]]
    df_sub_cama = pd.DataFrame({'CaMa': var}, index=time)
    df_sub_cama.index = df_sub_cama.index.strftime('%Y-%m')
    df['data'] = df['data'].drop(columns=['CaMa'])
    df['data'] = df['data'].join(df_sub_cama)
    var_o = df['data']['Value']
    var_s = df['data']['CaMa']
    # R
    _, _, R, _, _ = errlib.linergress(var_s, var_o)
    # NSE
    NSE = 1 - errlib.nse1(var_s, var_o, obj=False)
    # Q_sim
    Q_sim = np.nanmean(var_s) * 12

    # update data
    df['NSE'] = NSE
    df['Q_sim'] = Q_sim
    df['R'] = R
    data_up[id] = df

    if showfig:

        figsz = (4, 4)
        dpi = 100
        unit = 'Annual averaged discharge ($m^3/s$)'
        cmap = 'YlOrBr'
        fig = plt.figure(figsize=figsz, dpi=dpi)
        ax = fig.add_subplot(111)

        # xlim
        if ij[0] % 10 < 5 and ij[0] > 10:
            ylim1 = ij[0] // 10 * 10 - 9
        else:
            ylim1 = ij[0] // 10 * 10 + 1
        ylim2 = ylim1 + 20
        # ylim
        if ij[1] % 10 < 5 and ij[1] > 10:
            xlim1 = ij[1] // 10 * 10 - 9
        else:
            xlim1 = ij[1] // 10 * 10 + 1
        xlim2 = xlim1 + 20

        ave_cama = np.nanmean(data_cama, axis=0)
        maxv = np.nanmax(ave_cama[ylim1:ylim2, xlim1:xlim2])
        cmax = int(str(maxv)[0] + '0' * (len(str(int(maxv // 1))) - 1))
        clim = [0, int(str(cmax)[0] + '0' * (len(str(int(cmax // 1))) - 1))]

        norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
        ax.imshow(ave_cama[ylim1:ylim2, xlim1:xlim2], norm=norm, cmap=cmap)
        ax.set_xticks([0, 10, 19])
        ax.set_yticks([0, 10, 19])
        ax.set_xticklabels([ll[1] - (ij[1] - xlim1) * 0.5, ll[1] + ((ij[1] - xlim1) * 0.5 - (ylim2 - ij[0]) * 0.5) / 2, ll[1] + (xlim2 - ij[1]) * 0.5])
        ax.set_yticklabels([ll[0] + (ij[0] - ylim1) * 0.5, ll[0] + ((ij[0] - ylim1) * 0.5 - (ylim2 - ij[0]) * 0.5) / 2, ll[0] - (ylim2 - ij[0]) * 0.5])

        # original position of the gauge
        ax.plot(ij[1] - xlim1, ij[0] - ylim1, 'rx')

        # colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3.5%', pad=0.1)
        cbar = mpl.colorbar.ColorbarBase(cax, norm=norm, cmap=cmap, extend='max')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(unit, rotation=270)

        cbar.outline.set_visible(True)
        cbar.outline.set_edgecolor('grey')
        cbar.outline.set_linewidth(0.5)

        # set title
        ax.set_title(data_up[id]['Station'])

        # corrected position of the gauge
        ax.plot(ij_new[1] - xlim1, ij_new[0] - ylim1, 'bo')

    return data_up, criteria


def grdc_cama(data_grdc, data_cama, time, threshold=20000, exclude=[]):
    # p_VAL_RUNOFF.ipynb

    data_up = {}
    tablec = []
    for id in data_grdc.keys():
        if float(data_grdc[id]['Catchment_area']) > threshold and id not in exclude:
            # grdc
            df = data_grdc[id].copy()
            df_sub = df['data']
            if len(df_sub) == 0 or all(np.isnan(df_sub['Value'])):
                continue
            try:
                df['data'].index = df['data'].index.strftime('%Y-%m')
            except AttributeError:
                pass
            # cama
            ij = get_coor(float(data_grdc[id]['Latitude']), float(data_grdc[id]['Longitude']))
            var = data_cama[:, ij[0], ij[1]]
            df_sub_cama = pd.DataFrame({'CaMa': var}, index=time)
            df_sub_cama.index = df_sub_cama.index.strftime('%Y-%m')
            df['data'] = df_sub.join(df_sub_cama)
            var_o = df['data']['Value']
            var_s = df['data']['CaMa']
            _, _, R, _, _ = errlib.linergress(var_s, var_o)
            df = OrderedDict(df)
            # R
            df['R'] = R
            df.move_to_end('R', last=False)
            # Year
            df['Year'] = df_sub.index[0][:5] + df_sub.index[-1][:4]
            df.move_to_end('Year', last=False)
            # Q mean
            Q_obs = np.nanmean(var_o) * 12
            Q_sim = np.nanmean(var_s) * 12
            df['Q_obs'] = Q_obs
            df['Q_sim'] = Q_sim
            df.move_to_end('Q_obs', last=False)
            df.move_to_end('Q_sim', last=False)
            # NSE
            NSE = 1 - errlib.nse1(var_s, var_o, obj=False)
            df['NSE'] = NSE
            df.move_to_end('NSE', last=False)
            # append df
            data_up[id] = df
            # table
            row = []
            head = ['GRDC_No', 'River', 'Station', 'Longitude', 'Latitude', 'Catchment_area', 'Year', 'Q_obs', 'Q_sim', 'R', 'NSE']
            for key in head:
                row.append(df[key])
            tablec.append(row)
    table = pd.DataFrame(tablec, columns=head)
    return table, data_up


def grdc_table(data_up):
    tablec = []
    for id in data_up.keys():
        df = data_up[id].copy()
        row = []
        head = ['GRDC_No', 'River', 'Station', 'Longitude', 'Latitude', 'Catchment_area', 'Year', 'Q_obs', 'Q_sim', 'R', 'NSE']
        for key in head:
            row.append(df[key])
        tablec.append(row)
    table = pd.DataFrame(tablec, columns=head)
    return table


def domain_filter(data, domain):
    if type(domain) == list:
        domain = np.array(domain)
    domain = domain.astype('int')
    data[domain == 0] = np.nan

    return data


def domain_fill(data, domain, k=5):
    data = domain_filter(data, domain)

    sz = data.shape
    data_new = data.copy()

    for i in range(sz[0]):
        for j in range(sz[1]):
            if np.isnan(data_new[i, j]) and domain[i, j]:
                r_i = [i - k, i + k]
                r_j = [j - k, j + k]
                if i - k < 0:
                    r_i[0] = 0
                elif i + k > sz[1] - 1:
                    r_i[1] = sz[1] - 1
                if j - k < 0:
                    r_j[0] = 0
                elif j + k > sz[1] - 1:
                    r_j[1] = sz[1] - 1
                data_new[i, j] = np.nanmean(data_new[r_i[0]:r_i[1], r_j[0]:r_j[1]])

    return data_new


def domain_smooth(data, domain, k=5):
    data = domain_filter(data, domain)

    sz = data.shape
    data_new = data.copy()

    for i in range(sz[0]):
        for j in range(sz[1]):
            if domain[i, j]:
                r_i = [i - k, i + k]
                r_j = [j - k, j + k]
                if i - k < 0:
                    r_i[0] = 0
                elif i + k > sz[1] - 1:
                    r_i[1] = sz[1] - 1
                if j - k < 0:
                    r_j[0] = 0
                elif j + k > sz[1] - 1:
                    r_j[1] = sz[1] - 1
                data_new[i, j] = np.nanmean(data_new[r_i[0]:r_i[1], r_j[0]:r_j[1]])
    return data_new


def read_cru(file):
    rzsc = np.transpose(ascii(file))
    rzsc[rzsc == -9999] = np.nan
    rzsc = np.vstack((np.full([20, 720], np.nan), rzsc, np.full([68, 720], np.nan)))
    rzsc[rzsc < 10] = 10
    return rzsc


def read_chirps(file):
    rzsc = np.transpose(ascii(file))
    rzsc[rzsc == -9999] = np.nan
    rzsc = np.vstack((np.full([80, 720], np.nan), rzsc, np.full([80, 720], np.nan)))
    rzsc[rzsc < 10] = 10
    return rzsc


def cr(dat, figure_on=False):
    # change ratio
    if all(np.isnan(dat)):
        return np.nan, np.nan, np.nan, np.nan

    if type(dat) != np.ndarray:
        dat = np.array(dat)

    dat1 = np.arange(len(dat))
    iddat1 = np.isfinite(dat)
    degree = 1
    x = dat1[iddat1]
    y = dat[iddat1]
    fit = np.polyfit(x, y, degree)
    model = np.poly1d(fit)
    df = pd.DataFrame(columns=['y', 'x'])
    df['x'] = x
    df['y'] = y
    results = smf.ols(formula='y ~ model(x)', data=df).fit()

    fit_fn = np.poly1d(fit)
    if figure_on:
        plt.plot(dat1, fit_fn(dat1), 'k-')
        plt.plot(dat1, dat, 'go', ms=2)
        print(results.summary())

    # changes, mean of data, trend in percentage, p-value
    return fit[0] * len(dat1), np.nanmean(dat), (fit[0] * len(dat1)) / np.nanmean(dat) * 100, results.f_pvalue, fit_fn


def find_nearest(array, value):
    # Find nearest value in numpy array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def normalize(d):
    max_value = np.nanmax(d)
    min_value = np.nanmin(d)
    d_new = (d - min_value) / (max_value - min_value)
    return d_new


def movave(d, window, mode='nearest', axis=0):
    weights = np.ones(window)

    if type(d) == list:
        d = np.array(d)

    d1 = convolve1d(d, weights=weights, mode=mode, axis=axis) / window

    return d1


def movave_ds(ds, window):
    return xr.apply_ufunc(movave, ds, window, dask='allowed')


if __name__ == '__main__':
    # obj_setups = read_obj_setups('.././configs/obj_setups.txt')
    # print(obj_setups)
    # gof_setups = read_gof_setups('.././configs/gof_setups.txt')
    # print(gof_setups)
    # param_setups = read_param_setups('.././configs/param_setups.txt')
    # print(param_setups)
    var_id = 'pr: pr, tas: tas, tasmin: tasmin, tasmax: tasmax, rsds: rsds, rlds: rlds, wind: wind, rhs: rhs, pet: pet'
    file_syb = 'xx_ppp_ssss_eeee.nc4'
    p_name = 'gswp3'
    year_start = '1991'
    year_end = '2000'
    print(namestr(var_id, file_syb, p_name, year_start, year_end))
