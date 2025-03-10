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

from seppy.util import resample_df


logger = pooch.get_logger()
logger.setLevel("WARNING")


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
    #
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
    except (RuntimeError, IndexError):
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
    print('')

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
                if downloaded_file == []:
                    print('Trying download from CDAWeb...')
                    downloaded_file = Fido.fetch(result[0][i], path=path)  #, max_conn=max_conn)

    except (RuntimeError, IndexError):
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
    # cdf = cdflib.CDF(files[0])
    # df = _cdf2df_3d(cdf, "Epoch")
    df =_read_cdf_wind3dp(files[0])

    # read additional cdf files
    if len(files) > 1:
        for f in files[1:]:
            # cdf = cdflib.CDF(f)
            # t_df = _cdf2df_3d(cdf, "Epoch")
            t_df =_read_cdf_wind3dp(f)
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
        df = resample_df(df=df, resample=resample, pos_timestamp="center", origin="start")

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
        # from Table 3 of Wilson et al. 2021, https://doi.org/10.1029/2020RG000714
        # ∼30% ΔE/E => ΔE = 0.3*E
        if dataset in ['WI_SFSP_3DP', 'WI_SFPD_3DP', 'WI_SOSP_3DP', 'WI_SOPD_3DP']:
            delta_e = 0.3 * e_mean
        # ∼20% ΔE/E => ΔE = 0.2*E
        elif dataset in ['WI_ELSP_3DP', 'WI_ELPD_3DP', 'WI_EHSP_3DP', 'WI_EHPD_3DP']:
            delta_e = 0.2 * e_mean
        e_low = e_mean - delta_e
        e_high = e_mean + delta_e
        energies = pd.concat([e_mean, delta_e, e_low, e_high], axis=1, keys=['mean_E', 'DE', 'lower_E', 'upper_E'])
        energies['Bins_Text']= np.around(e_low/1e3, 2).astype('string') +' - '+ np.around(e_high/1e3, 2).astype('string') + ' keV'

        meta = {'channels_dict_df': energies,
                'ENERGY_UNITS': metacdf.varattsget('ENERGY')['UNITS'],
                'FLUX_UNITS': metacdf.varattsget('FLUX')['UNITS'],
                'FLUX_FILLVAL': metacdf.varattsget('FLUX')['FILLVAL'],
                }

        # for SFSP, SOSP, SFPD, SFSP:
        try:
            meta['APPROX_ENERGY_LABELS'] = metacdf.varget('APPROX_ENERGY_LABELS')
            meta['FLUX_LABELS'] = metacdf.varget('FLUX_ENERGY_LABL')
        except:
            pass

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


"""
Modification of sunpy's read_cdf function to allow loading of 3-dimensional variables from a cdf file. Adjusted only for Wind/3DP cdf files. Skipping units.
This function is copied from sunpy under the terms of the BSD 2-Clause licence. See licenses/SUNPY_LICENSE.rst
"""


def _read_cdf_wind3dp(fname, ignore_vars=[]):
    """
    Read a CDF file that follows the ISTP/IACG guidelines.

    Parameters
    ----------
    fname : path-like
        Location of single CDF file to read.

    Returns
    -------
    list[GenericTimeSeries]
        A list of time series objects, one for each unique time index within
        the CDF file.

    References
    ----------
    Space Physics Guidelines for CDF https://spdf.gsfc.nasa.gov/sp_use_of_cdf.html
    """
    import astropy.units as u
    from cdflib.epochs import CDFepoch
    from packaging.version import Version
    from sunpy import log
    from sunpy.timeseries import GenericTimeSeries
    from sunpy.util.exceptions import warn_user
    cdf = cdflib.CDF(str(fname))
    # Extract the time varying variables
    cdf_info = cdf.cdf_info()
    meta = cdf.globalattsget()
    if hasattr(cdflib, "__version__") and Version(cdflib.__version__) >= Version("1.0.0"):
        all_var_keys = cdf_info.rVariables + cdf_info.zVariables
    else:
        all_var_keys = cdf_info['rVariables'] + cdf_info['zVariables']
    var_attrs = {key: cdf.varattsget(key) for key in all_var_keys}

    # Get keys that depend on time
    # var_keys = [var for var in var_attrs if 'DEPEND_0' in var_attrs[var] and var_attrs[var]['DEPEND_0'] is not None]
    # Manually define keys that depend on time for Wind/3DP cdf files, as they don't follow the standard
    var_keys = all_var_keys

    # Get unique time index keys
    # time_index_keys = sorted(set([var_attrs[var]['DEPEND_0'] for var in var_keys]))
    # Manually define time index key for Wind/3DP cdf files, as they don't follow the standard
    time_index_keys = [var_keys.pop(var_keys.index('Epoch'))]

    all_ts = []
    # For each time index, construct a GenericTimeSeries
    for index_key in time_index_keys:
        try:
            index = cdf.varget(index_key)
        except ValueError:
            # Empty index for cdflib >= 0.3.20
            continue
        # TODO: use to_astropy_time() instead here when we drop pandas in timeseries
        index = CDFepoch.to_datetime(index)
        df = pd.DataFrame(index=pd.DatetimeIndex(name=index_key, data=index))
        # units = {}

        # for var_key in sorted(var_keys):
        for var_key in var_keys:
            if var_key in ignore_vars:
                continue  # leave for-loop, skipping var_key

            attrs = var_attrs[var_key]
            # Skip the following check for Wind/3DP cdf files, as they don't follow the standard
            # # If this variable doesn't depend on this index, continue
            # if attrs['DEPEND_0'] != index_key:
            #     continue

            # Get data
            if hasattr(cdflib, "__version__") and Version(cdflib.__version__) >= Version("1.0.0"):
                var_last_rec = cdf.varinq(var_key).Last_Rec
            else:
                var_last_rec = cdf.varinq(var_key)['Last_Rec']
            if var_last_rec == -1:
                log.debug(f'Skipping {var_key} in {fname} as it has zero elements')
                continue

            data = cdf.varget(var_key)

            # Skip the following code block for Wind/3DP cdf files, as they don't follow the standard
            # # Set fillval values to NaN
            # # It would be nice to properley mask these values to work with
            # # non-floating point (ie. int) dtypes, but this is not possible with pandas
            # if np.issubdtype(data.dtype, np.floating):
            #     data[data == attrs['FILLVAL']] = np.nan

            # Skip the following code block for Wind/3DP cdf files, as they don't follow the standard
            # # Get units
            # if 'UNITS' in attrs:
            #     unit_str = attrs['UNITS']
            #     try:
            #         unit = u.Unit(unit_str)
            #     except ValueError:
            #         if unit_str in _known_units:
            #             unit = _known_units[unit_str]
            #         else:
            #             warn_user(f'astropy did not recognize units of "{unit_str}". '
            #                       'Assigning dimensionless units. '
            #                       'If you think this unit should not be dimensionless, '
            #                       'please raise an issue at https://github.com/sunpy/sunpy/issues')
            #             unit = u.dimensionless_unscaled
            # else:
            #     warn_user(f'No units provided for variable "{var_key}". '
            #               'Assigning dimensionless units.')
            #     unit = u.dimensionless_unscaled

            if data.ndim > 3:
                # Skip data with dimensions >= 3 and give user warning
                warn_user(f'The variable "{var_key}" has been skipped because it has more than 3 dimensions, which is unsupported.')
            elif data.ndim == 3:
                # Multiple columns, give each column a unique label.
                # Numbering hard-corded to Wind/3DP data!
                for j in range(data.T.shape[0]):
                    for i, col in enumerate(data.T[j, :, :]):
                        var_key_mod = var_key + f'_E{j}'
                        df[var_key_mod + f'_P{i}'] = col
                        # units[var_key_mod + f'_{i}'] = unit
            elif data.ndim == 2:
                # Multiple columns, give each column a unique label
                for i, col in enumerate(data.T):
                    df[var_key + f'_{i}'] = col
                    # units[var_key + f'_{i}'] = unit
            else:
                # Single column
                df[var_key] = data
                # units[var_key] = unit

        # all_ts.append(GenericTimeSeries(data=df, units=units, meta=meta))

    # if not len(all_ts):
    #     log.debug(f'No data found in file {fname}')
    return df  # all_ts[0].to_dataframe()
