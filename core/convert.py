"""
Functions for unit conversion .
"""

import math


def celsius2kelvin(celsius):
    """
    Convert temperature in degrees Celsius to degrees Kelvin.

    :param celsius: Degrees Celsius
    :return: Degrees Kelvin
    :rtype: float
    """
    return celsius + 273.15


def kelvin2celsius(kelvin):
    """
    Convert temperature in degrees Kelvin to degrees Celsius.

    :param kelvin: Degrees Kelvin
    :return: Degrees Celsius
    :rtype: float
    """
    return kelvin - 273.15


def deg2rad(degrees):
    """
    Convert angular degrees to radians

    :param degrees: Value in degrees to be converted.
    :return: Value in radians
    :rtype: float
    """
    return degrees * (math.pi / 180.0)


def rad2deg(radians):
    """
    Convert radians to angular degrees

    :param radians: Value in radians to be converted.
    :return: Value in angular degrees
    :rtype: float
    """
    return radians * (180.0 / math.pi)


def mjmd2wm(mjmd):
    """
    Convert MJ m-2 d-1 to W m-2

    :param mjmd: Value in MJ m-2 to be converted.
    :return: Value in W m-2
    :rtype: float
    MJ = 1e6 J
    d = 86400 s
    J s-1 = W
    """
    return mjmd * 1e6 / 86400


def wm2mjmd(wm):
    """
    Convert W m-2 to MJ m-2 d-1

    :param mjmd: Value in W m-2 to be converted.
    :return: Value in MJ m-2
    :rtype: float
    MJ = 1e6 J
    s-1 = 86400 d-1
    W = J s-1
    """
    return wm * 86400 / 1e6


def kgms2mmd(kgms):
    """
    Convert kg m-2 s-1 to mm d-1

    :param kgms: Value in kg m-2 s-1 to be converted.
    :return: Value in mm d-1
    :rtype: float
    """
    return kgms * 24 * 60 * 60


def mmd2kgms(mmd):
    """
    Convert mm d-1 to kg m-2 s-1

    :param kgms: Value in mm to be converted.
    :return: Value in kg m-2 s-1
    :rtype: float
    """
    return mmd / 24 / 60 / 60


def wm2mmd(wm):
    """
    Convert W m-2 to mm d-1

    :param wm: Value in W m-2 to be converted.
    :return: Value in mm d-1
    :rtype: float
    """
    return wm * 0.03527


def mmd2wm(mmd):
    """
    Convert mm d-1 to W m-2

    :param wm: Value in mm to be converted.
    :return: Value in W m-2
    :rtype: float
    """
    return mmd / 0.03527
