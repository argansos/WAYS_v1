#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-29 16:35:00
# @Author  : Ganquan Mao (ganquan.mao@icloud.com)
# @Link    : https://argansos.github.io
# @Version : 3.6

'''
s - simulation
o - observation
'''


import numpy as np
import pandas as pd
from scipy import stats


def filter_na(s, o):
    df = pd.DataFrame({'s': s, 'o': o})
    return df.dropna()


def rmse(s, o):
    df = filter_na(s, o)
    if df.empty:
        rmse = np.NaN
        nrmse = np.NaN
    else:
        rmse = ((df.s - df.o) ** 2).mean() ** .5
        nrmse = rmse / df.o.mean()
    return rmse, nrmse


def pc_bias(s, o):
    # Percent Bias

    df = filter_na(s, o)

    return 100.0 * sum(df.s - df.o) / sum(df.o)


def NS(s, o):
    # Nash Sutcliffe efficiency coefficient
    df = filter_na(s, o)

    if df.empty:
        return np.NaN
    else:
        return 1 - sum((df.s - df.o)**2) / sum((df.o - np.mean(df.o))**2)


def nse1(s, o, obj=True):
    # 1 - NSE
    df = filter_na(s, o)
    if df.empty:
        return np.NaN
    else:
        if obj:
            return sum((df.s - df.o) ** 2)
        else:
            return sum((df.s - df.o) ** 2) / sum((df.o - np.mean(df.o)) ** 2)


def linergress(s, o):
    df = filter_na(s, o)
    if df.empty:
        return [np.NaN] * 5
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(df.s, df.o)
    return slope, intercept, r_value, p_value, std_err


def cv(x):
    x = x[~np.isnan(x)]

    return stats.variation(x)
