#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ganquan Mao (ganquan.mao@icloud.com)
# @Link    : https://argansos.github.io
# @Version : 3.6


import pprint
import numpy as np
from pyhdf.SD import SD, SDC


def hdf4(file, var_name='', print_out=False):
    # read a MODIS HDF4 file
    dat = SD(file, SDC.READ)
    datasets_dic = dat.datasets()
    if len(var_name) == 0:
        for idx, sds in enumerate(datasets_dic.keys()):
            print(idx, sds)
        return ''

    sds_obj = dat.select(var_name)  # select sds

    data = sds_obj.get()  # get sds data
    data = data.astype('float')

    data[data == sds_obj.attributes()['_FillValue']] = np.nan
    data /= sds_obj.attributes()['scale_factor']

    if print_out:
        pprint.pprint(sds_obj.attributes())

    return data
