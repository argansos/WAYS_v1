#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ganquan Mao (ganquan.mao@icloud.com)
# @Link    : https://argansos.github.io
# @Version : 3.6


import math


def daylength(dateinfo, lat, method='forsythe'):
    # calculate the daylength
    # Herbert Glarner's formula
    # http://herbert.gandraxa.com/length_of_day.xml
    # Forsythe's formula
    # W. Forsythe, E. Rykiel, R. Stahl, H. Wu and R. Schoolfield, "A model comparison for daylength as a function of latitude and day of year", Ecological Modelling, vol. 80, no. 1, pp. 87-95, 1995.
    # dateinfo - pandas Timestamp or datetime.date

    day = dateinfo.timetuple().tm_yday

    if method == 'forsythe':
        # revolution angle from day of the year
        theta = 0.2163108 + 2 * math.atan(0.9671396 * math.tan(0.00860 * (day - 186)))

        # sun declination angle
        P = math.asin(0.39795 * math.cos(theta))

        # daylength (plus twilight)
        p = 0.8333  # sunrise/sunset is when the top of the sun is apparently even with horizon
        x = (math.sin(p * math.pi / 180) + math.sin(lat * math.pi / 180) * math.sin(P)) / (math.cos(lat * math.pi / 180) * math.cos(P))
        # cut the tail which excess the range
        if x > 1:
            x = 1
        elif x < -1:
            x = -1

        # daylength in hour
        hours = 24 - (24 / math.pi) * math.acos(x)

    elif method == 'herbert':
        axis = 23.439 * math.pi / 180

        j = math.pi / 182.625

        m = 1 - math.tan(lat * math.pi / 180) * math.tan(axis * math.cos(j * day))

        # saturated value for artics
        if m > 2:
            m = 2
        elif m < 0:
            m = 0

        b = math.acos(1 - m) / math.pi

        # daylength in hour
        hours = b * 24

    return hours


def stress_tmin(tmin):
    # stress function of minimum temperature
    tmin_lowb = 271.15  # K
    tmin_uppb = 278.15  # K
    tmin = tmin + 273.17  # C to K

    # formula
    if tmin <= tmin_lowb:
        strees_ix = 0
    elif tmin >= tmin_uppb:
        strees_ix = 1
    else:
        strees_ix = (tmin - tmin_lowb) / (tmin_uppb - tmin_lowb)

    return strees_ix


def stress_dayl(dayl):
    # stress function of day length
    dayl_lowb = 36000  # s
    dayl_uppb = 39600  # s
    dayl = dayl * 3600  # hour to second

    # formula
    if dayl <= dayl_lowb:
        stress_ix = 0
    elif dayl >= dayl_uppb:
        stress_ix = 1
    else:
        stress_ix = (dayl - dayl_lowb) / (dayl_uppb - dayl_lowb)

    return stress_ix


def stress_moist(srz, rzsc, method='matsumoto'):
    # stress function of soil moisture
    c_rz = 0.07

    # formula
    if srz <= 0:
        stress_ix = 0
    elif srz >= rzsc:
        stress_ix = 1
    else:
        if method == 'matsumoto':
            stress_ix = (srz * (rzsc + c_rz)) / (rzsc * (srz + c_rz))
        elif method == 'feddes':
            stress_ix = srz / 0.5 * rzsc
        elif method == 'genuchten':
            stress_ix = srz / rzsc
        else:
            raise ValueError('method is not defined!')

    return stress_ix


def igs(dateinfo, tmin, srz, lat, rzsc):
    # growing-season index

    # stress of min temperature
    stmin = stress_tmin(tmin)

    # stress of day length
    dayl = daylength(dateinfo, lat)
    sdayl = stress_dayl(dayl)

    # stress of moisture
    smoist = stress_moist(srz, rzsc)

    return stmin * sdayl * smoist


def lai(igs_21, lu):
    # calculating the leaf area index
    # lu    Land Use                            RV      LAImax      LAImin
    # 00    Water                               NA      0           0
    # 01    Evergreen Needleleaf forest         20      5.5         2
    # 02    Evergreen Broadleaf forest          60      5.5         2
    # 03    Deciduous Needleleaf forest         60      5           1
    # 04    Deciduous Broadleaf forest          2       5.5         1
    # 05    Mixed forest                        60      5           1
    # 06    Closed shrublands                   2       1.5         0.5
    # 07    Open shrublands                     2       1.5         0.5
    # 08    Woody savannas                      10      2           0.5
    # 09    Savannas                            10      2           0.5
    # 10    Grasslands                          2       2           0.5
    # 11    Permanent wetlands                  NA      4           1
    # 12    Croplands                           2       3.5         0.5
    # 13    Urban and built-up                  NA      1           0.1
    # 14    Cropland/Natural vegetation mosaic  2       3.5         0.5
    # 15    Snow and ice                        NA      0           0
    # 16    Barren or sparsely vegetated        2       0.1         0.01
    # 254   Unclassified                        NA      NA          NA
    # 255   Fill Value                          NA      NA          NA

    # maximum and minimum value of lai for different land use
    lai_max = [0, 5.5, 5.5, 5, 5.5, 5, 1.5, 1.5, 2, 2, 2, 4, 3.5, 1, 3.5, 0, 0.1]
    lai_min = [0, 2, 2, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.1, 0.5, 0, 0.01]

    # lai value
    lai_val = lai_min[lu] + igs_21 * (lai_max[lu] - lai_min[lu])

    return lai_val
