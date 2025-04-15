# Licensed under a 3-clause BSD style license - see LICENSE.rst

import copy
import os
import warnings

import cdflib
import numpy as np
import pandas as pd
import sunpy

from packaging.version import Version
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries

from seppy.util import resample_df

# Not needed atm as units are skipped in the modified read_cdf
# if hasattr(sunpy, "__version__") and Version(sunpy.__version__) >= Version("5.0.0"):
#     from sunpy.io._cdf import read_cdf, _known_units
# else:
#     from sunpy.io.cdf import read_cdf, _known_units


def _fillval_nan(data, fillval):
    try:
        data[data == fillval] = np.nan
    except ValueError:
        # This happens if we try and assign a NaN to an int type
        pass
    return data


def _get_cdf_vars(cdf):
    # Get list of all the variables in an open CDF file
    var_list = []
    cdf_info = cdf.cdf_info()
    for attr in list(cdf_info.keys()):
        if 'variable' in attr.lower() and len(cdf_info[attr]) > 0:
            for var in cdf_info[attr]:
                var_list += [var]

    return var_list


# def _cdf2df_3d_psp(cdf, index_key, dtimeindex=True, ignore=None, include=None):
#     """
#     Converts a cdf file to a pandas dataframe.
#     Note that this only works for 1 dimensional data, other data such as
#     distribution functions or pitch angles will not work properly.
#     Parameters
#     ----------
#     cdf : cdf
#         Opened CDF file.
#     index_key : str
#         The CDF key to use as the index in the output DataFrame.
#     dtimeindex : bool
#         If ``True``, the DataFrame index is parsed as a datetime.
#         Default is ``True``.
#     ignore : list
#         In case a CDF file has columns that are unused / not required, then
#         the column names can be passed as a list into the function.
#     include : str, list
#         If only specific columns of a CDF file are desired, then the column
#         names can be passed as a list into the function. Should not be used
#         with ``ignore``.
#     Returns
#     -------
#     df : :class:`pandas.DataFrame`
#         Data frame with read in data.
#     """
#     if include is not None:
#         if ignore is not None:
#             raise ValueError('ignore and include are incompatible keywords')
#         if isinstance(include, str):
#             include = [include]
#         if index_key not in include:
#             include.append(index_key)

#     # Extract index values
#     index_info = cdf.varinq(index_key)
#     if index_info['Last_Rec'] == -1:
#         warnings.warn(f"No records present in CDF file {cdf.cdf_info()['CDF'].name}")
#         return_df = pd.DataFrame()
#     else:
#         index = cdf.varget(index_key)
#         try:
#             # If there are multiple indexes, take the first one
#             # TODO: this is just plain wrong, there should be a way to get all
#             # the indexes out
#             index = index[...][:, 0]
#         except IndexError:
#             pass

#         if dtimeindex:
#             index = cdflib.epochs.CDFepoch.breakdown(index, to_np=True)
#             index_df = pd.DataFrame({'year': index[:, 0],
#                                      'month': index[:, 1],
#                                      'day': index[:, 2],
#                                      'hour': index[:, 3],
#                                      'minute': index[:, 4],
#                                      'second': index[:, 5],
#                                      'ms': index[:, 6],
#                                      })
#             # Not all CDFs store pass milliseconds
#             try:
#                 index_df['us'] = index[:, 7]
#                 index_df['ns'] = index[:, 8]
#             except IndexError:
#                 pass
#             index = pd.DatetimeIndex(pd.to_datetime(index_df), name='Time')
#         data_dict = {}
#         npoints = len(index)

#         var_list = _get_cdf_vars(cdf)
#         keys = {}
#         # Get mapping from each attr to sub-variables
#         for cdf_key in var_list:
#             if ignore:
#                 if cdf_key in ignore:
#                     continue
#             elif include:
#                 if cdf_key not in include:
#                     continue
#             if cdf_key == 'Epoch':
#                 keys[cdf_key] = 'Time'
#             else:
#                 keys[cdf_key] = cdf_key
#         # Remove index key, as we have already used it to create the index
#         keys.pop(index_key)
#         # Remove keys for data that doesn't have the right shape to load in CDF
#         # Mapping of keys to variable data
#         vars = {}
#         for cdf_key in keys.copy():
#             try:
#                 vars[cdf_key] = cdf.varget(cdf_key)
#             except ValueError:
#                 vars[cdf_key] = ''
#         for cdf_key in keys:
#             var = vars[cdf_key]
#             if type(var) is np.ndarray:
#                 key_shape = var.shape
#                 if len(key_shape) == 0 or key_shape[0] != npoints:
#                     vars.pop(cdf_key)
#             else:
#                 vars.pop(cdf_key)

#         # Loop through each key and put data into the dataframe
#         for cdf_key in vars:
#             df_key = keys[cdf_key]
#             # Get fill value for this key
#             # First catch string FILLVAL's
#             if type(cdf.varattsget(cdf_key)['FILLVAL']) is str:
#                 fillval = cdf.varattsget(cdf_key)['FILLVAL']
#             else:
#                 try:
#                     fillval = float(cdf.varattsget(cdf_key)['FILLVAL'])
#                 except KeyError:
#                     fillval = np.nan

#             if isinstance(df_key, list):
#                 for i, subkey in enumerate(df_key):
#                     data = vars[cdf_key][...][:, i]
#                     data = _fillval_nan(data, fillval)
#                     data_dict[subkey] = data
#             else:
#                 # If ndims is 1, we just have a single column of data
#                 # If ndims is 2, have multiple columns of data under same key
#                 # If ndims is 3, have multiple columns of data under same key, with 2 sub_keys (e.g., energy and pitch-angle)
#                 key_shape = vars[cdf_key].shape
#                 ndims = len(key_shape)
#                 if ndims == 1:
#                     data = vars[cdf_key][...]
#                     data = _fillval_nan(data, fillval)
#                     data_dict[df_key] = data
#                 elif ndims == 2:
#                     for i in range(key_shape[1]):
#                         data = vars[cdf_key][...][:, i]
#                         data = _fillval_nan(data, fillval)
#                         data_dict[f'{df_key}_{i}'] = data
#                 elif ndims == 3:
#                     for i in range(key_shape[2]):
#                         for j in range(key_shape[1]):
#                             data = vars[cdf_key][...][:, j, i]
#                             data = _fillval_nan(data, fillval)
#                             data_dict[f'{df_key}_E{i}_P{j}'] = data
#         return_df = pd.DataFrame(index=index, data=data_dict)

#     return return_df


def psp_isois_load(dataset, startdate, enddate, epilo_channel='F', epilo_threshold=None, path=None, resample=None, all_columns=False):
    """
    Downloads CDF files via SunPy/Fido from CDAWeb for CELIAS, EPHIN, ERNE onboard SOHO
    Parameters
    ----------
    dataset : {str}
        Name of PSP dataset:
            - 'PSP_ISOIS-EPIHI_L2-HET-RATES60'
            - 'PSP_ISOIS-EPIHI_L2-HET-RATES3600' (higher coverage than 'RATES60' before mid-2021)
            - 'PSP_ISOIS-EPIHI_L2-LET1-RATES60'
            - 'PSP_ISOIS-EPIHI_L2-LET2-RATES60'
            - 'PSP_ISOIS-EPILO_L2-PE'
            - 'PSP_ISOIS-EPILO_L2-IC'
    startdate, enddate : {datetime or str}
        Datetime object (e.g., dt.date(2021,12,31) or dt.datetime(2021,4,15)) or "standard"
        datetime string (e.g., "2021/04/15") (enddate must always be later than startdate)
    epilo_channel : string
        'E', 'F', 'G' (for 'EPILO PE'), or 'C', 'D', 'P', 'R', 'T' (for 'EPILO IC').
        EPILO chan, by default 'F'
    epilo_threshold : {int or float}, optional
        Replace ALL flux/countrate values above 'epilo_threshold' with np.nan, by default None.
        Only works for Electron count rates in 'PSP_ISOIS-EPILO_L2-PE' dataset
    path : {str}, optional
        Local path for storing downloaded data, by default None
    resample : {str}, optional
        resample frequency in format understandable by Pandas, e.g. '1min', by default None
    all_columns : {boolean}, optional
        Whether to return all columns of the datafile for EPILO (or skip
        usually unneeded columns for better performance), by default False
    Returns
    -------
    df : {Pandas dataframe}
        See links above for the different datasets for a description of the dataframe columns
    energies_dict : {dictionary}
        Dictionary containing energy information.
        NOTE: For EPIHI energy values are only loaded from the first day of the interval!
        For EPILO energy values are the mean of the whole loaded interval.
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
                downloaded_file = Fido.fetch(result[0][i], path=path, max_conn=1)

        # loading for EPIHI
        if dataset.split('-')[1] == 'EPIHI_L2':
            # downloaded_files = Fido.fetch(result, path=path, max_conn=1)
            # downloaded_files.sort()
            data = TimeSeries(downloaded_files, concatenate=True)
            df = data.to_dataframe()
            # df = read_cdf(downloaded_files[0])

            # # reduce data frame to only H_Flux, H_Uncertainty, Electron_Counts, and Electron_Rate.
            # # There is no Electron_Uncertainty, maybe one could use at least the Poission error from Electron_Counts for that.
            # # df = df.filter(like='H_Flux') + df.filter(like='H_Uncertainty') + df.filter(like='Electrons')
            # if dataset.split('-')[2].upper() == 'HET':
            #     if dataset.split('-')[3] == 'RATES60':
            #         selected_cols = ["A_H_Flux", "B_H_Flux", "A_H_Uncertainty", "B_H_Uncertainty", "A_Electrons", "B_Electrons"]
            #     if dataset.split('-')[3] == 'RATES3600':
            #         selected_cols = ["A_H_Flux", "B_H_Flux", "A_H_Uncertainty", "B_H_Uncertainty", "A_Electrons", "B_Electrons"]
            # if dataset.split('-')[2].upper() == 'LET1':
            #     selected_cols = ["A_H_Flux", "B_H_Flux", "A_H_Uncertainty", "B_H_Uncertainty", "A_Electrons", "B_Electrons"]
            # if dataset.split('-')[2].upper() == 'LET2':
            #     selected_cols = ["A_H_Flux", "B_H_Flux", "A_H_Uncertainty", "B_H_Uncertainty", "A_Electrons", "B_Electrons"]
            # df = df[df.columns[df.columns.str.startswith(tuple(selected_cols))]]
            # raise Warning(f"{dataset} is not fully suppported, only proton and electron data will be processed!")

            cdf = cdflib.CDF(downloaded_files[0])

            # remove this (i.e. following line) when sunpy's read_cdf is updated,
            # and FILLVAL will be replaced directly, see
            # https://github.com/sunpy/sunpy/issues/5908
            # df = df.replace(cdf.varattsget('A_H_Flux')['FILLVAL'], np.nan)
            # 4 Apr 2023: previous 1 lines removed because they are taken care of with sunpy
            # 4.1.0:
            # https://docs.sunpy.org/en/stable/whatsnew/changelog.html#id7
            # https://github.com/sunpy/sunpy/pull/5956

            # get info on energies and units
            energies_dict = {"H_ENERGY":
                             cdf['H_ENERGY'],
                             "H_ENERGY_DELTAPLUS":
                             cdf['H_ENERGY_DELTAPLUS'],
                             "H_ENERGY_DELTAMINUS":
                             cdf['H_ENERGY_DELTAMINUS'],
                             "H_ENERGY_LABL":
                             cdf['H_ENERGY_LABL'],
                             "Electrons_ENERGY":
                             cdf['Electrons_ENERGY'],
                             "Electrons_ENERGY_DELTAPLUS":
                             cdf['Electrons_ENERGY_DELTAPLUS'],
                             "Electrons_ENERGY_DELTAMINUS":
                             cdf['Electrons_ENERGY_DELTAMINUS'],
                             "Electrons_ENERGY_LABL":
                             cdf['Electrons_ENERGY_LABL']
                             }
            try:
                energies_dict["H_FLUX_UNITS"] = cdf.varattsget('A_H_Flux')['UNITS']
                energies_dict["Electrons_Rate_UNITS"] = cdf.varattsget('A_Electrons_Rate')['UNITS']
            except ValueError:
                try:
                    energies_dict["H_FLUX_UNITS"] = cdf.varattsget('C_H_Flux')['UNITS']
                    energies_dict["Electrons_Rate_UNITS"] = cdf.varattsget('C_Electrons_Rate')['UNITS']
                except ValueError:
                    raise Warning(f"Can't obtain UNITS from metadata. Possibly an unsupported dataset is loaded!")

        # loading for EPILO
        if dataset.split('-')[1] == 'EPILO_L2':
            if dataset[-2:] == 'PE':
                species_str = 'Electron'
            elif dataset[-2:] == 'IC':
                species_str = 'H'

            if len(downloaded_files) > 0:
                if all_columns:
                    ignore = []
                else:
                    ignore = [f'Epoch_Chan{epilo_channel}_DELTA', f'HCI_Chan{epilo_channel}', f'HCI_Lat_Chan{epilo_channel}', f'HCI_Lon_Chan{epilo_channel}',
                              f'HCI_R_Chan{epilo_channel}', f'HGC_Lat_Chan{epilo_channel}', f'HGC_Lon_Chan{epilo_channel}', f'HGC_R_Chan{epilo_channel}',
                              f'{species_str}_Chan{epilo_channel}_Energy_LABL', f'{species_str}_Counts_Chan{epilo_channel}', f'RTN_Chan{epilo_channel}']
                    # ignore = ['Epoch_ChanP_DELTA', 'HCI_ChanP', 'HCI_Lat_ChanP', 'HCI_Lon_ChanP', 'HCI_R_ChanP', 'HGC_Lat_ChanP', 'HGC_Lon_ChanP', 'HGC_R_ChanP', 'H_ChanP_Energy', 'H_ChanP_Energy_DELTAMINUS', 'H_ChanP_Energy_DELTAPLUS', 'H_ChanP_Energy_LABL', 'H_CountRate_ChanP', 'H_Counts_ChanP', 'H_Flux_ChanP', 'H_Flux_ChanP_DELTA', 'PA_ChanP', 'Quality_Flag_ChanP', 'RTN_ChanP', 'SA_ChanP

                # read 0th cdf file
                # # cdf = cdflib.CDF(downloaded_files[0])
                # # df = _cdf2df_3d_psp(cdf, f"Epoch_Chan{epilo_channel.upper()}", ignore=ignore)
                df = _read_cdf_psp(downloaded_files[0], f"Epoch_Chan{epilo_channel.upper()}", ignore_vars=ignore)

                # read additional cdf files
                if len(downloaded_files) > 1:
                    for f in downloaded_files[1:]:
                        # # cdf = cdflib.CDF(f)
                        # # t_df = _cdf2df_3d_psp(cdf, f"Epoch_Chan{epilo_channel.upper()}", ignore=ignore)
                        t_df = _read_cdf_psp(f, f"Epoch_Chan{epilo_channel.upper()}", ignore_vars=ignore)
                        df = pd.concat([df, t_df])

                # columns of returned df for EPILO PE
                # -----------------------------------
                # PA_ChanF_0 to PA_ChanF_7
                # SA_ChanF_0 to SA_ChanF_7
                # Electron_ChanF_Energy_E0_P0 to Electron_ChanF_Energy_E47_P7
                # Electron_ChanF_Energy_DELTAMINUS_E0_P0 to Electron_ChanF_Energy_DELTAMINUS_E47_P7
                # Electron_ChanF_Energy_DELTAPLUS_E0_P0 to Electron_ChanF_Energy_DELTAPLUS_E47_P7
                # Electron_CountRate_ChanF_E0_P0 to Electron_CountRate_ChanF_E47_P7
                energies_dict = {}
                for k in [f'{species_str}_Chan{epilo_channel.upper()}_Energy_E',
                          f'{species_str}_Chan{epilo_channel.upper()}_Energy_DELTAMINUS',
                          f'{species_str}_Chan{epilo_channel.upper()}_Energy_DELTAPLUS']:
                    energies_dict[k] = df[df.columns[df.columns.str.startswith(k)]].mean()
                    df.drop(df.columns[df.columns.str.startswith(k)], axis=1, inplace=True)
                # rename energy column (removing trailing '_E')
                energies_dict[f'{species_str}_Chan{epilo_channel.upper()}_Energy'] = energies_dict.pop(f'{species_str}_Chan{epilo_channel.upper()}_Energy_E')

                # replace outlier data points above given threshold with np.nan
                # note: df.where(cond, np.nan) replaces all values where the cond is NOT fullfilled with np.nan
                # following Pandas Dataframe work is not too elegant, but works...
                if epilo_threshold:
                    # create new dataframe of FLUX columns only with removed outliers
                    df2 = df.filter(like='Electron_CountRate_').where(df.filter(like='Electron_CountRate_') <= epilo_threshold, np.nan)
                    # drop these FLUX columns from original dataframe
                    flux_cols = df.filter(like='Electron_CountRate_').columns
                    df.drop(labels=flux_cols, axis=1, inplace=True)
                    # add cleaned new FLUX columns to original dataframe
                    df = pd.concat([df2, df], axis=1)
            else:
                df = ''
                energies_dict = ''

        if isinstance(resample, str):
            df = resample_df(df=df, resample=resample, pos_timestamp="center", origin="start")

    except (RuntimeError, IndexError):
        print(f'Unable to obtain "{dataset}" data!')
        downloaded_files = []
        df = pd.DataFrame()
        energies_dict = []
    return df, energies_dict


def calc_av_en_flux_PSP_EPIHI(df, energies, en_channel, species, instrument, viewing):
    """
    This function averages the flux of several energy channels into a combined energy channel
    channel numbers counted from 0

    So far only works for EPIHI-HET

    Parameters
    ----------
    df : pd.DataFrame DataFrame containing HET data
        DataFrame containing PSP data
    energies : dict
        Energy dict returned from psp_load
    en_channel : int or list
        energy channel number(s) to be used
    species : string
        'e', 'electrons', 'p', 'i', 'protons', 'ions'
    instrument : string
        'het'
    viewing : string
        'A', 'B'

    Returns
    -------
    pd.DataFrame
        flux_out: contains channel-averaged flux
    """
    if instrument.lower() == 'het':
        if species.lower() in ['e', 'electrons']:
            species_str = 'Electrons'
            flux_key = 'Electrons_Rate'
        if species.lower() in ['p', 'protons', 'i', 'ions', 'h']:
            species_str = 'H'
            flux_key = 'H_Flux'
    en_str = energies[f'{species_str}_ENERGY_LABL']
    if type(en_channel) == list:
        energy_low = en_str[en_channel[0]].flat[0].split('-')[0]
        energy_up = en_str[en_channel[-1]].flat[0].split('-')[-1]
        en_channel_string = energy_low + '-' + energy_up

        DE = energies[f'{species_str}_ENERGY_DELTAPLUS']+energies[f'{species_str}_ENERGY_DELTAMINUS']

        if len(en_channel) > 2:
            raise Exception('en_channel must have len 2 or less!')
        if len(en_channel) == 2:
            try:
                df = df[df.columns[df.columns.str.startswith(f'{viewing.upper()}_{flux_key}')]]
            except (AttributeError, KeyError):
                None
            for bins in np.arange(en_channel[0], en_channel[-1]+1):
                if bins == en_channel[0]:
                    I_all = df[f'{viewing.upper()}_{flux_key}_{bins}'] * DE[bins]
                else:
                    I_all = I_all + df[f'{viewing.upper()}_{flux_key}_{bins}'] * DE[bins]
            DE_total = np.sum(DE[(en_channel[0]):(en_channel[-1]+1)])
            flux_out = pd.DataFrame({'flux': I_all/DE_total}, index=df.index)
        else:
            en_channel = en_channel[0]
            flux_out = pd.DataFrame({'flux': df[f'{viewing.upper()}_{flux_key}_{en_channel}']}, index=df.index)
    else:
        flux_out = pd.DataFrame({'flux': df[f'{viewing.upper()}_{flux_key}_{en_channel}']}, index=df.index)
        en_channel_string = en_str[en_channel].flat[0]
    # replace multiple whitespaces with single ones
    en_channel_string = ' '.join(en_channel_string.split())
    return flux_out, en_channel_string


def calc_av_en_flux_PSP_EPILO(df, en_dict, en_channel, species, mode, chan, viewing):
    """
    This function averages the flux of several energy channels (and viewing directions) into a combined energy channel.
    channel numbers counted from 0

    So far only works for EPILO PE chanF electrons

    Parameters
    ----------
    df : pd.DataFrame DataFrame containing HET data
        DataFrame containing PSP data
    energies : dict
        Energy dict returned from psp_load
    en_channel : int or list
        energy channel number(s) to be used
    species : string
        'e', 'electrons'
    mode : string
        'pe' or 'ic'. EPILO mode
    chan : string
        'E', 'F', 'G', 'P', 'T'. EPILO chan
    viewing : int or list
        EPILO viewing. 0 to 7 for electrons; 0 to 79 for ions
        (ions 70-79 correspond to electrons 7, i.e., the electron wedges are
        split up into 10 viewings for ions)

    Returns
    -------
    pd.DataFrame
        flux_out: contains channel-averaged flux
    """

    if mode.lower() == 'pe':
        if species.lower() in ['e', 'electrons']:
            species_str = 'Electron'
            flux_key = 'Electron_CountRate'
        # if species.lower() in ['p', 'protons', 'i', 'ions', 'h']:
        #     species_str = 'H'
        #     flux_key = 'H_Flux'
    elif mode.lower() == 'ic':
        # if species.lower() in ['e', 'electrons']:
        #     species_str = 'Electrons'
        #     flux_key = 'Electrons_Rate'
        if species.lower() in ['p', 'protons', 'i', 'ions', 'h']:
            species_str = 'H'
            flux_key = 'H_Flux'
    if type(en_channel) == int:
        en_channel = [en_channel]
    if type(viewing) == int:
        viewing = [viewing]

    df_out = pd.DataFrame()
    flux_out_all = {}
    en_channel_string_all = []
    for view in viewing:
        if type(en_channel) == list:
            # energy = en_dict[f'{species_str}_Chan{chan}_Energy'].filter(like=f'_P{view}').values
            energy = en_dict[f'{species_str}_Chan{chan}_Energy'][en_dict[f'{species_str}_Chan{chan}_Energy'].keys().str.endswith(f'_P{view}')].values
            # energy_low = energy - en_dict[f'{species_str}_Chan{chan}_Energy_DELTAMINUS'].filter(like=f'_P{view}').values
            energy_low = energy - en_dict[f'{species_str}_Chan{chan}_Energy_DELTAMINUS'][en_dict[f'{species_str}_Chan{chan}_Energy_DELTAMINUS'].keys().str.endswith(f'_P{view}')].values
            # energy_high = energy + en_dict[f'{species_str}_Chan{chan}_Energy_DELTAPLUS'].filter(like=f'_P{view}').values
            energy_high = energy + en_dict[f'{species_str}_Chan{chan}_Energy_DELTAPLUS'][en_dict[f'{species_str}_Chan{chan}_Energy_DELTAPLUS'].keys().str.endswith(f'_P{view}')].values
            DE = en_dict[f'{species_str}_Chan{chan}_Energy_DELTAMINUS'].filter(like=f'_P{view}').values + en_dict[f'{species_str}_Chan{chan}_Energy_DELTAPLUS'].filter(like=f'_P{view}').values

            # build energy string of combined channel
            en_channel_string = np.round(energy_low[en_channel[0]], 1).astype(str) + ' - ' + np.round(energy_high[en_channel[-1]], 1).astype(str) + ' keV'

            # select view direction
            # df = df.filter(like=f'_P{view}')

            if len(en_channel) > 2:
                raise Exception("en_channel must have length 2 or less! Define first and last channel to use (don't list all of them)")
            if len(en_channel) == 2:
                # try:
                #     df = df[df.columns[df.columns.str.startswith(f'{view.upper()}_{flux_key}')]]
                #     # df = df[df.columns[df.columns.str.startswith(f'{flux_key}_Chan{chan}_')]]
                # except (AttributeError, KeyError):
                #     None
                for bins in np.arange(en_channel[0], en_channel[-1]+1):
                    if bins == en_channel[0]:
                        I_all = df[f"{flux_key}_Chan{chan}_E{bins}_P{view}"] * DE[bins]
                    else:
                        I_all = I_all + df[f"{flux_key}_Chan{chan}_E{bins}_P{view}"] * DE[bins]
                DE_total = np.sum(DE[(en_channel[0]):(en_channel[-1]+1)])
                flux_out = pd.DataFrame({f'viewing_{view}': I_all/DE_total}, index=df.index)
            if len(en_channel) == 1:
                en_channel = en_channel[0]
                flux_out = pd.DataFrame({f'viewing_{view}': df[f"{flux_key}_Chan{chan}_E{en_channel}_P{view}"]}, index=df.index)

            df_out = pd.concat([df_out, flux_out], axis=1)

        # calculate mean of all viewings:
        df_out2 = pd.DataFrame({'flux': df_out.mean(axis=1, skipna=True)}, index=df_out.index)
        en_channel_string_all.append(en_channel_string)

    # check if not all elements of en_channel_string_all are the same:
    if len(en_channel_string_all) != en_channel_string_all.count(en_channel_string_all[0]):
        print("You are combining viewing directions that have different energies. This is strongly advised against!")
        print(en_channel_string_all)
        return df_out2, en_channel_string_all[0]
    else:
        return df_out2, en_channel_string_all[0]


psp_load = copy.copy(psp_isois_load)


"""
Modification of sunpy's read_cdf function to allow skipping of reading variables from a cdf file.
This function is copied from sunpy under the terms of the BSD 2-Clause licence. See licenses/SUNPY_LICENSE.rst
"""


def _read_cdf_psp(fname, index_key, ignore_vars=[]):
    """
    Read a CDF file that follows the ISTP/IACG guidelines.

    Parameters
    ----------
    fname : path-like
        Location of single CDF file to read.
    index_key : str
        The CDF key to use as the index in the output DataFrame.
        For example, index_key='Epoch_ChanP'
    ignore_vars : list
        In case a CDF file has columns that are unused / not required, then
        the column names can be passed as a list into the function.

    Returns
    -------
    DataFrame
        A Pandas DataFrame for the time index defined by index_key.

    References
    ----------
    Space Physics Guidelines for CDF https://spdf.gsfc.nasa.gov/sp_use_of_cdf.html
    """
    import astropy.units as u
    from cdflib.epochs import CDFepoch
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
    var_keys = [var for var in var_attrs if 'DEPEND_0' in var_attrs[var] and var_attrs[var]['DEPEND_0'] is not None]

    # # Get unique time index keys
    # time_index_keys = sorted(set([var_attrs[var]['DEPEND_0'] for var in var_keys]))

    # all_ts = []
    # # For each time index, construct a GenericTimeSeries
    # for index_key in time_index_keys:
    #     try:
    #         index = cdf.varget(index_key)
    #     except ValueError:
    #         # Empty index for cdflib >= 0.3.20
    #         continue

    # Only for selected index_key:
    index = cdf.varget(index_key)

    # TODO: use to_astropy_time() instead here when we drop pandas in timeseries
    index = CDFepoch.to_datetime(index)
    # df = pd.DataFrame(index=pd.DatetimeIndex(name=index_key, data=index))
    units = {}
    df_dict = {}

    for var_key in var_keys:
        if var_key in ignore_vars:
            continue  # leave for-loop, skipping var_key

        attrs = var_attrs[var_key]
        # If this variable doesn't depend on this index, continue
        if attrs['DEPEND_0'] != index_key:
            continue

        # Get data
        if hasattr(cdflib, "__version__") and Version(cdflib.__version__) >= Version("1.0.0"):
            var_last_rec = cdf.varinq(var_key).Last_Rec
        else:
            var_last_rec = cdf.varinq(var_key)['Last_Rec']
        if var_last_rec == -1:
            log.debug(f'Skipping {var_key} in {fname} as it has zero elements')
            continue

        data = cdf.varget(var_key)

        # Set fillval values to NaN
        # It would be nice to properley mask these values to work with
        # non-floating point (ie. int) dtypes, but this is not possible with pandas
        if np.issubdtype(data.dtype, np.floating):
            data[data == attrs['FILLVAL']] = np.nan

        # Skip all units :-(
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
        #                         'Assigning dimensionless units. '
        #                         'If you think this unit should not be dimensionless, '
        #                         'please raise an issue at https://github.com/sunpy/sunpy/issues')
        #             unit = u.dimensionless_unscaled
        # else:
        #     warn_user(f'No units provided for variable "{var_key}". '
        #                 'Assigning dimensionless units.')
        #     unit = u.dimensionless_unscaled

        if data.ndim > 3:
            # Skip data with dimensions >= 3 and give user warning
            warn_user(f'The variable "{var_key}" has been skipped because it has more than 3 dimensions, which is unsupported.')
        elif data.ndim == 3:
            # Multiple columns, give each column a unique label.
            for j in range(data.T.shape[0]):
                for i, col in enumerate(data.T[j, :, :]):
                    # var_key_mod = var_key+'_E'+str(j).rjust(2, '0')
                    var_key_mod = var_key+f'_E{j}'
                    # df[var_key_mod + '_P'+str(i).rjust(2, '0')] = col
                    df_dict[var_key_mod + f'_P{i}'] = col
                    # units[var_key_mod + f'_{i}'] = unit
        elif data.ndim == 2:
            # Multiple columns, give each column a unique label
            for i, col in enumerate(data.T):
                df_dict[var_key + f'_{i}'] = col
                # units[var_key + f'_{i}'] = unit
        else:
            # Single column
            df_dict[var_key] = data
            # units[var_key] = unit

        df = pd.DataFrame(df_dict, index=pd.DatetimeIndex(name=index_key, data=index))

    # all_ts.append(GenericTimeSeries(data=df, units=units, meta=meta))

    # if not len(all_ts):
    if not len(df):
        log.debug(f'No data found in file {fname}')
    return df  # all_ts
