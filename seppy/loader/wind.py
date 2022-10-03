# Licensed under a 3-clause BSD style license - see LICENSE.rst

import cdflib
import copy
import datetime as dt
import numpy as np
import os
import pandas as pd
import pooch
import requests
import sunpy
import warnings

from sunpy.net import Fido
from sunpy.net import attrs as a


def _date2str(date):
    year = str(date)[0:4]
    month = str(date)[4:6]
    day = str(date)[6:8]
    return year+'/'+month+'/'+day


def _cdf2df_3d(cdf, index_key, dtimeindex=True, badvalues=None,
               ignore=None, include=None):
    """
    Converts a cdf file to a pandas dataframe.
    Note that this only works for 1 dimensional data, other data such as
    distribution functions or pitch angles will not work properly.
    Parameters
    ----------
    cdf : cdf
        Opened CDF file.
    index_key : str
        The CDF key to use as the index in the output DataFrame.
    dtimeindex : bool
        If ``True``, the DataFrame index is parsed as a datetime.
        Default is ``True``.
    badvalues : dict, list
        Deprecated.
    ignore : list
        In case a CDF file has columns that are unused / not required, then
        the column names can be passed as a list into the function.
    include : str, list
        If only specific columns of a CDF file are desired, then the column
        names can be passed as a list into the function. Should not be used
        with ``ignore``.
    Returns
    -------
    df : :class:`pandas.DataFrame`
        Data frame with read in data.
    """
    if badvalues is not None:
        warnings.warn('The badvalues argument is decprecated, as bad values '
                      'are now automatically recognised using the FILLVAL CDF '
                      'attribute.', DeprecationWarning)
    if include is not None:
        if ignore is not None:
            raise ValueError('ignore and include are incompatible keywords')
        if isinstance(include, str):
            include = [include]
        if index_key not in include:
            include.append(index_key)

    # Extract index values
    index_info = cdf.varinq(index_key)
    if index_info['Last_Rec'] == -1:
        warnings.warn('No records present in CDF file')

    index = cdf.varget(index_key)
    try:
        # If there are multiple indexes, take the first one
        # TODO: this is just plain wrong, there should be a way to get all
        # the indexes out
        index = index[...][:, 0]
    except IndexError:
        pass

    if dtimeindex:
        index = cdflib.epochs.CDFepoch.breakdown(index, to_np=True)
        index_df = pd.DataFrame({'year': index[:, 0],
                                 'month': index[:, 1],
                                 'day': index[:, 2],
                                 'hour': index[:, 3],
                                 'minute': index[:, 4],
                                 'second': index[:, 5],
                                 'ms': index[:, 6],
                                 })
        # Not all CDFs store pass milliseconds
        try:
            index_df['us'] = index[:, 7]
            index_df['ns'] = index[:, 8]
        except IndexError:
            pass
        index = pd.DatetimeIndex(pd.to_datetime(index_df), name='Time')
    data_dict = {}
    npoints = len(index)

    var_list = _get_cdf_vars(cdf)
    keys = {}
    # Get mapping from each attr to sub-variables
    for cdf_key in var_list:
        if ignore:
            if cdf_key in ignore:
                continue
        elif include:
            if cdf_key not in include:
                continue
        if cdf_key == 'Epoch':
            keys[cdf_key] = 'Time'
        else:
            keys[cdf_key] = cdf_key
    # Remove index key, as we have already used it to create the index
    keys.pop(index_key)
    # Remove keys for data that doesn't have the right shape to load in CDF
    # Mapping of keys to variable data
    vars = {cdf_key: cdf.varget(cdf_key) for cdf_key in keys.copy()}
    for cdf_key in keys:
        var = vars[cdf_key]
        if type(var) is np.ndarray:
            key_shape = var.shape
            if len(key_shape) == 0 or key_shape[0] != npoints:
                vars.pop(cdf_key)
        else:
            vars.pop(cdf_key)

    # Loop through each key and put data into the dataframe
    for cdf_key in vars:
        df_key = keys[cdf_key]
        # Get fill value for this key
        try:
            fillval = float(cdf.varattsget(cdf_key)['FILLVAL'])
        except KeyError:
            fillval = np.nan

        if isinstance(df_key, list):
            for i, subkey in enumerate(df_key):
                data = vars[cdf_key][...][:, i]
                data = _fillval_nan(data, fillval)
                data_dict[subkey] = data
        else:
            # If ndims is 1, we just have a single column of data
            # If ndims is 2, have multiple columns of data under same key
            # If ndims is 3, have multiple columns of data under same key, with 2 sub_keys (e.g., energy and pitch-angle)
            key_shape = vars[cdf_key].shape
            ndims = len(key_shape)
            if ndims == 1:
                data = vars[cdf_key][...]
                data = _fillval_nan(data, fillval)
                data_dict[df_key] = data
            elif ndims == 2:
                for i in range(key_shape[1]):
                    data = vars[cdf_key][...][:, i]
                    data = _fillval_nan(data, fillval)
                    data_dict[f'{df_key}_{i}'] = data
            elif ndims == 3:
                for i in range(key_shape[2]):
                    for j in range(key_shape[1]):
                        data = vars[cdf_key][...][:, j, i]
                        data = _fillval_nan(data, fillval)
                        data_dict[f'{df_key}_E{i}_P{j}'] = data

    return pd.DataFrame(index=index, data=data_dict)


def _get_cdf_vars(cdf):
    # Get list of all the variables in an open CDF file
    var_list = []
    cdf_info = cdf.cdf_info()
    for attr in list(cdf_info.keys()):
        if 'variable' in attr.lower() and len(cdf_info[attr]) > 0:
            for var in cdf_info[attr]:
                var_list += [var]

    return var_list


def _fillval_nan(data, fillval):
    try:
        data[data == fillval] = np.nan
    except ValueError:
        # This happens if we try and assign a NaN to an int type
        pass
    return data


def _download_metafile(dataset, path=None):
    """
    Download master cdf file from cdaweb for 'dataset'
    """
    if not path:
        path = sunpy.config.get('downloads', 'sample_dir')
    base_url = 'https://spdf.gsfc.nasa.gov/pub/software/cdawlib/0MASTERS/'
    fname = dataset.lower() + '_00000000_v01.cdf'
    url = base_url + fname
    try:
        downloaded_file = pooch.retrieve(url=url, known_hash=None, fname=fname, path=path, progressbar=True)
    except ModuleNotFoundError:
        downloaded_file = pooch.retrieve(url=url, known_hash=None, fname=fname, path=path, progressbar=False)
    return downloaded_file


def wind3dp_download_fido(dataset, startdate, enddate, path=None, max_conn=5):
    """
    Downloads Wind/3DP CDF files via SunPy/Fido from CDAWeb

    Parameters
    ----------
    dataset : {str}
        Name of Wind/3DP dataset:
        - 'WI_SFSP_3DP': Electron omnidirectional fluxes 27 keV - 520 keV, often
            at 24 sec
        - 'WI_SFPD_3DP': Electron energy-angle distributions 27 keV to 520 keV,
            often at 24 sec
        - 'WI_SOSP_3DP': Proton omnidirectional fluxes 70 keV - 6.8 MeV, often
            at 24 sec
        - 'WI_SOPD_3DP': Proton energy-angle distributions 70 keV - 6.8 MeV,
            often at 24 sec
    startdate, enddate : {datetime or str}
        Datetime object (e.g., dt.date(2021,12,31) or dt.datetime(2021,4,15)) or
        "standard" datetime string (e.g., "2021/04/15") (enddate must always be
        later than startdate)
    path : {str}, optional
        Local path for storing downloaded data, by default None
    max_conn : {int}, optional
        The number of parallel download slots used by Fido.fetch, by default 5

    Returns
    -------
    List of downloaded files
    """
    trange = a.Time(startdate, enddate)
    cda_dataset = a.cdaweb.Dataset(dataset)
    try:
        result = Fido.search(trange, cda_dataset)
        filelist = [i[0].split('/')[-1] for i in result.show('URL')[0]]
        filelist.sort()
        if path is None:
            filelist = [sunpy.config.get('downloads', 'download_dir') + os.sep + file for file in filelist]
        elif type(path) is str:
            filelist = [path + os.sep + f for f in filelist]
        downloaded_files = filelist

        for i, f in enumerate(filelist):
            if os.path.exists(f) and os.path.getsize(f) == 0:
                os.remove(f)
            if not os.path.exists(f):
                downloaded_file = Fido.fetch(result[0][i], path=path, max_conn=max_conn)
        # downloaded_files = Fido.fetch(result, path=path, max_conn=max_conn)
        # downloaded_files.sort()
    except RuntimeError:
        print(f'Unable to obtain "{dataset}" data for {startdate}-{enddate}!')
        downloaded_files = []
    return downloaded_files


def wind3dp_single_download(file, path=None):
    """
    Download a single Wind/3DP level 2 data file from SRL Berkeley to local path

    Parameters
    ----------
    file : str
        file to be downloaded, e.g. 'wi_sfsp_3dp_20220602_v01.cdf'
    path : str
        local path where the files will be stored
    Returns
    -------
    downloaded_file : str
        full local path to downloaded file
    """

    # add a OS-specific '/' to end end of 'path'
    if path:
        if not path[-1] == os.sep:
            path = f'{path}{os.sep}'
    else:
        path = sunpy.config.get('downloads', 'download_dir') + os.sep

    data = file.split('_')[1]  # e.g. 'sfsp'
    year = file.split('_')[3][:4]
    base = f"https://sprg.ssl.berkeley.edu/wind3dp/data/wi/3dp/{data}/{year}/"

    url = base+'/'+file

    try:
        downloaded_file = pooch.retrieve(url=url, known_hash=None, fname=file, path=path, progressbar=True)
    except ModuleNotFoundError:
        downloaded_file = pooch.retrieve(url=url, known_hash=None, fname=file, path=path, progressbar=False)
    except requests.HTTPError:
        print(f'No corresponding data found at {url}')
        downloaded_file = []

    return downloaded_file


def wind3dp_download(dataset, startdate, enddate, path=None, **kwargs):
    """
    Downloads Wind/3DP CDF files via SunPy/Fido from CDAWeb

    Parameters
    ----------
    dataset : {str}
        Name of Wind/3DP dataset:
        - 'WI_SFSP_3DP': Electron omnidirectional fluxes 27 keV - 520 keV, often
            at 24 sec
        - 'WI_SFPD_3DP': Electron energy-angle distributions 27 keV to 520 keV,
            often at 24 sec
        - 'WI_SOSP_3DP': Proton omnidirectional fluxes 70 keV - 6.8 MeV, often
            at 24 sec
        - 'WI_SOPD_3DP': Proton energy-angle distributions 70 keV - 6.8 MeV,
            often at 24 sec
    startdate, enddate : {datetime or str}
        Datetime object (e.g., dt.date(2021,12,31) or dt.datetime(2021,4,15)) or
        "standard" datetime string (e.g., "2021/04/15") (enddate must always be
        later than startdate)
    path : {str}, optional
        Local path for storing downloaded data, by default None

    Returns
    -------
    List of downloaded files
    """

    trange = a.Time(startdate, enddate)
    cda_dataset = a.cdaweb.Dataset(dataset)
    try:
        result = Fido.search(trange, cda_dataset)
        filelist = [i[0].split('/')[-1] for i in result.show('URL')[0]]
        filelist.sort()
        files = filelist
        if path is None:
            filelist = [sunpy.config.get('downloads', 'download_dir') + os.sep + file for file in filelist]
        elif type(path) is str:
            filelist = [path + os.sep + f for f in filelist]
        downloaded_files = filelist

        for i, f in enumerate(filelist):
            if os.path.exists(f) and os.path.getsize(f) == 0:
                os.remove(f)
            if not os.path.exists(f):
                # downloaded_file = Fido.fetch(result[0][i], path=path, max_conn=max_conn)
                downloaded_file = wind3dp_single_download(files[i], path=path)

    except RuntimeError:
        print(f'Unable to obtain "{dataset}" data for {startdate}-{enddate}!')
        downloaded_files = []
    return downloaded_files


def _wind3dp_load(files, resample="1min", threshold=None):
    if isinstance(resample, str):
        try:
            _ = pd.Timedelta(resample)
        except ValueError:
            raise Warning(f"Your 'resample' option of [{resample}] doesn't seem to be a proper Pandas frequency!")
    # try:
    # read 0th cdf file
    cdf = cdflib.CDF(files[0])
    df = _cdf2df_3d(cdf, "Epoch")

    # read additional cdf files
    if len(files) > 1:
        for f in files[1:]:
            cdf = cdflib.CDF(f)
            t_df = _cdf2df_3d(cdf, "Epoch")
            df = pd.concat([df, t_df])

    # replace bad data with np.nan:
    df = df.replace(-np.inf, np.nan)

    # replace outlier data points above given threshold with np.nan
    # note: df.where(cond, np.nan) replaces all values where the cond is NOT fullfilled with np.nan
    # following Pandas Dataframe work is not too elegant, but works...
    if threshold:
        # create new dataframe of FLUX columns only with removed outliers
        df2 = df.filter(like='FLUX_').where(df.filter(like='FLUX_') <= threshold, np.nan)
        # drop these FLUX columns from original dataframe
        flux_cols = df.filter(like='FLUX_').columns
        df.drop(labels=flux_cols, axis=1, inplace=True)
        # add cleaned new FLUX columns to original dataframe
        df = pd.concat([df2, df], axis=1)

    if isinstance(resample, str):
        df = df.resample(resample).mean()
        df.index = df.index + pd.tseries.frequencies.to_offset(pd.Timedelta(resample)/2)
    return df
    # except:
    #     raise Exception(f"Problem while loading CDF file! Delete downloaded file(s) {files} and try again. Sometimes this is enough to solve the problem.")


def wind3dp_load(dataset, startdate, enddate, resample="1min", multi_index=True,
                 path=None, threshold=None, **kwargs):
    """
    Load-in data for Wind/3DP instrument. Provides released data obtained by
    SunPy through CDF files from CDAWeb. Returns data as Pandas dataframe.

    Parameters
    ----------
    dataset : {str}
        Name of Wind/3DP dataset:
        - 'WI_SFSP_3DP': Electron omnidirectional fluxes 27 keV - 520 keV, often
            at 24 sec
        - 'WI_SFPD_3DP': Electron energy-angle distributions 27 keV to 520 keV,
            often at 24 sec
        - 'WI_SOSP_3DP': Proton omnidirectional fluxes 70 keV - 6.8 MeV, often
            at 24 sec
        - 'WI_SOPD_3DP': Proton energy-angle distributions 70 keV - 6.8 MeV,
            often at 24 sec
    startdate, enddate : {datetime or str}
        Datetime object (e.g., dt.date(2021,12,31) or dt.datetime(2021,4,15)) or
        "standard" datetime string (e.g., "2021/04/15") (enddate must always be
        later than startdate)
    resample : {str}, optional
        Frequency to which the original data (~24 seconds) is resamepled. Pandas
        frequency (e.g., '1min' or '1h') or None, by default "1min"
    multi_index : {bool}, optional
        Provide output for pitch-angle resolved data as Pandas Dataframe with
        multiindex, by default True
    path : {str}, optional
        Local path for storing downloaded data, by default None
    threshold : {int or float}, optional
        Replace all FLUX values above 'threshold' with np.nan, by default None

    Returns
    -------
    _type_
        _description_
    """
    files = wind3dp_download(dataset, startdate, enddate, path)
    if len(files) > 0:
        df = _wind3dp_load(files, resample, threshold)

        # download master file from CDAWeb
        path_to_metafile = _download_metafile(dataset, path=path)

        # open master file from CDAWeb as cdf
        metacdf = cdflib.CDF(path_to_metafile)

        e_mean = df.filter(like='ENERGY_').mean()
        # ∼30% ΔE/E => ΔE = 0.3*E
        # from Table 3 of Wilson et al. 2021, https://doi.org/10.1029/2020RG000714
        delta_e = 0.3 * e_mean
        e_low = e_mean - delta_e
        e_high = e_mean + delta_e
        energies = pd.concat([e_mean, delta_e, e_low, e_high], axis=1, keys=['mean_E', 'DE', 'lower_E', 'upper_E'])
        energies['Bins_Text']= np.around(e_low/1e3, 2).astype('string') +' - '+ np.around(e_high/1e3, 2).astype('string') + ' keV'

        meta = {'channels_dict_df': energies,
                'APPROX_ENERGY_LABELS': metacdf.varget('APPROX_ENERGY_LABELS'),
                'ENERGY_UNITS': metacdf.varattsget('ENERGY')['UNITS'],
                'FLUX_UNITS': metacdf.varattsget('FLUX')['UNITS'],
                'FLUX_FILLVAL': metacdf.varattsget('FLUX')['FILLVAL'],
                'FLUX_LABELS': metacdf.varget('FLUX_ENERGY_LABL'),
                }

        # create multi-index data frame of flux
        if multi_index:
            if dataset == 'WI_SFPD_3DP' or dataset == 'WI_SOPD_3DP':
                no_channels = len(df[df.columns[df.columns.str.startswith("ENERGY")]].columns)
                t_df = [''] * no_channels
                multi_keys = np.append([f"FLUX_E{i}" for i in range(no_channels)],
                                       df.drop(df.columns[df.columns.str.startswith(f"FLUX_")], axis=1).columns,
                                       )
                for i in range(no_channels):
                    t_df[i] = df[df.columns[df.columns.str.startswith(f"FLUX_E{i}")]]
                t_df.extend([df[col] for col in df.drop(df.columns[df.columns.str.startswith(f"FLUX_")], axis=1).columns.values])
                df = pd.concat(t_df, axis=1, keys=multi_keys)
            else:
                print('')
                print('Multi-index function only available (and necessary) for pitch-angle resolved fluxes. Skipping.')
    else:
        df = []
        meta = ''
    return df, meta


wind_load = copy.copy(wind3dp_load)
