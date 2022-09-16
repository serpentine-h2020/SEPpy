# Licensed under a 3-clause BSD style license - see LICENSE.rst

import datetime as dt
import os

import sunpy
from solo_epd_loader import epd_load
from solo_epd_loader import epd_load as solo_load
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries


def _date2str(date):
    year = str(date)[0:4]
    month = str(date)[4:6]
    day = str(date)[6:8]
    return year+'/'+month+'/'+day


def mag_load(startdate, enddate, level='l2', data_type='normal', frame='rtn', path=None):
    """
    Load SolO/MAG data

    Load-in data for Solar Orbiter/MAG sensor. Supports level 2 and low latency
    data provided by CDAWeb. Optionally downloads missing
    data directly. Returns data as Pandas dataframe.

    Parameters
    ----------
    startdate, enddate : {datetime, str, or int}
        Datetime object (e.g., dt.date(2021,12,31) or dt.datetime(2021,4,15)),
        "standard" datetime string (e.g., "2021/04/15") or integer of the form
        yyyymmdd with empty positions filled with zeros, e.g. '20210415'
        (enddate must always be later than startdate)
    level : {'l2', 'll'}, optional
        Defines level of data product: level 2 ('l2') or low-latency ('ll').
        By default 'l2'.
    data_type : {'normal', 'normal-1-minute', or 'burst'}, optional
        By default 'normal'.
    frame : {'rtn', 'srf', or 'vso'}, optional
        Coordinate frame of MAG data. By default 'rtn'.
    path : {str}, optional
        Local path for storing downloaded data, by default None

    Returns
    -------
    Pandas dataframe with fluxes and errors in 'particles / (s cm^2 sr MeV)'
    """
    if data_type == 'normal-1-minute' and frame == 'srf':
        raise Exception("For SRF frame only 'normal' or 'burst' data type available!")

    if data_type == 'normal-1-min':
        data_type = 'normal-1-minute'

    if level == 'll' or level == 'LL':
        level = 'll02'
        data_id = 'SOLO_'+level.upper()+'_MAG'
    else:
        data_id = 'SOLO_'+level.upper()+'_MAG-'+frame.upper()+'-'+data_type.upper()

    if isinstance(startdate, int):
        startdate = _date2str(startdate)
    if isinstance(enddate, int):
        enddate = _date2str(enddate)

    trange = a.Time(startdate, enddate)
    dataset = a.cdaweb.Dataset(data_id)
    result = Fido.search(trange, dataset)
    filelist = [i[0].split('/')[-1] for i in result.show('URL')[0]]
    filelist.sort()
    if path is None:
        filelist = [sunpy.config.get('downloads', 'download_dir') + os.sep + file for file in filelist]
    elif type(path) is str:
        filelist = [path + os.sep + f for f in filelist]

    for i, f in enumerate(filelist):
        if os.path.exists(f) and os.path.getsize(f) == 0:
            os.remove(f)
        if not os.path.exists(f):
            downloaded_file = Fido.fetch(result[0][i], path=path)
    # files = Fido.fetch(result, path=path)

    solo_mag = TimeSeries(filelist, concatenate=True)
    df_solo_mag = solo_mag.to_dataframe()
    return df_solo_mag


# VSO, LL02 not working
