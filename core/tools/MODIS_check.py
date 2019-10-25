#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ganquan Mao (ganquan.mao@icloud.com)
# @Link    : https://argansos.github.io
# @Version : 3.6


import os
import glob
import requests
import datetime
from bs4 import BeautifulSoup


def listwebDIR(url, pre='200'):
    # list the directories in a certain website
    page = requests.get(url).text
    # print(page)
    soup = BeautifulSoup(page, 'html.parser')
    DIR = [node.get('href')[:-1] for node in soup.find_all('a') if node.get('href').startswith(pre)]
    return DIR


def listwebHDFn(url, ext='hdf'):
    # return the number of hdf files in a certain website
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    HDF = [node.get('href')[:-1] + '/' for node in soup.find_all('a') if node.get('href').endswith(ext)]
    return len(HDF)


def nameindoy(date_str):
    # chang the y_m_d based datestr to a doy based datestr
    y, m, d = [int(x) for x in date_str.split('.')]
    doy = datetime.date(y, m, d).timetuple().tm_yday
    return str(y) + str(doy).zfill(3)


def listnof(o_dir):
    # return the number of the files in a certain path
    files = glob.glob(o_dir)
    return len(files)


def listdirHDFn(path, date_str):
    # return the number of the hdf files in a certain path
    path = os.path.join(path, 'MOD09A1.A' + nameindoy(date_str) + '*')
    return listnof(path)


if __name__ == '__main__':
    # path = "/Volumes/Liberty/Data/MODIS09A1"
    path = "/Volumes/TANK/DATA/Administrator/MODIS09A1"
    url = 'https://e4ftl01.cr.usgs.gov/MOLT/MOD09A1.006'
    pre = '2001'
    cmd = []
    for datedir in listwebDIR(url, pre):
        sub_url = os.path.join(url, datedir)
        if listwebHDFn(sub_url) != listdirHDFn(path, datedir):
            # str_p = 'wget -c -r -np -nd -k -L -p -A.hdf ' + sub_url  + '/ -P /Volumes/Liberty/Data/MODIS09A1'
            str_p = sub_url + '/'
            cmd.append(str_p)
    with open('filelist.txt', mode='wt') as myfile:
        myfile.write('\n'.join(line for line in cmd))
