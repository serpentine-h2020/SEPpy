# Licensed under a 3-clause BSD style license - see LICENSE.rst

import cdflib
import datetime as dt
import glob
import numpy as np
import os
import pandas as pd
import pooch
import requests
import sunpy
import warnings

from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries


def resample_df(df, resample, pos_timestamp='center'):
    """
    Resample Pandas Dataframe
    """
    try:
        df = df.resample(resample).mean()
        if pos_timestamp == 'start':
            df.index = df.index
        else:
            df.index = df.index + pd.tseries.frequencies.to_offset(pd.Timedelta(resample)/2)
        # if pos_timestamp == 'stop' or pos_timestamp == 'end':
        #     df.index = df.index + pd.tseries.frequencies.to_offset(pd.Timedelta(resample))
    except ValueError:
        raise ValueError(f"Your 'resample' option of [{resample}] doesn't seem to be a proper Pandas frequency!")
    return df


def _get_metadata(dataset, path_to_cdf):
    """
    Get meta data from single cdf file
    So far only manually for 'SOHO_ERNE-HED_L2-1MIN' and 'SOHO_ERNE-LED_L2-1MIN'
    """
    metadata = []
    cdf = cdflib.CDF(path_to_cdf)
    if dataset=='SOHO_ERNE-HED_L2-1MIN' or dataset=='SOHO_ERNE-LED_L2-1MIN':
        if dataset=='SOHO_ERNE-HED_L2-1MIN':
            m = 'H'
        if dataset=='SOHO_ERNE-LED_L2-1MIN':
            m = 'L'
        metadata = {'He_E_label': cdf.varget('He_E_label')[0],
                    'He_energy': cdf.varget('He_energy'),
                    'He_energy_delta': cdf.varget('He_energy_delta'),
                    f'A{m}_LABL': cdf.varattsget(f'A{m}')['LABLAXIS'],
                    f'A{m}_UNITS': cdf.varattsget(f'A{m}')['UNITS'],
                    f'A{m}_FILLVAL': cdf.varattsget(f'A{m}')['FILLVAL'],
                    'P_E_label': cdf.varget('P_E_label')[0],
                    'P_energy': cdf.varget('P_energy'),
                    'P_energy_delta': cdf.varget('P_energy_delta'),
                    f'P{m}_LABL': cdf.varattsget(f'P{m}')['LABLAXIS'],
                    f'P{m}_UNITS': cdf.varattsget(f'P{m}')['UNITS'],
                    f'P{m}_FILLVAL': cdf.varattsget(f'P{m}')['FILLVAL'],
                    }

        channels_dict_df_He = pd.DataFrame(cdf.varget('He_E_label')[0], columns=['ch_strings'])
        channels_dict_df_He['lower_E'] = cdf.varget("He_energy")-cdf.varget("He_energy_delta")
        channels_dict_df_He['upper_E'] = cdf.varget("He_energy")+cdf.varget("He_energy_delta")
        channels_dict_df_He['DE'] = cdf.varget("He_energy_delta")
        # channels_dict_df_He['mean_E'] = np.sqrt(channels_dict_df_He['upper_E'] * channels_dict_df_He['lower_E'])
        channels_dict_df_He['mean_E'] = cdf.varget("He_energy")

        channels_dict_df_p = pd.DataFrame(cdf.varget('P_E_label')[0], columns=['ch_strings'])
        channels_dict_df_p['lower_E'] = cdf.varget("P_energy")-cdf.varget("P_energy_delta")
        channels_dict_df_p['upper_E'] = cdf.varget("P_energy")+cdf.varget("P_energy_delta")
        channels_dict_df_p['DE'] = cdf.varget("P_energy_delta")
        # channels_dict_df_p['mean_E'] = np.sqrt(channels_dict_df_p['upper_E'] * channels_dict_df_p['lower_E'])
        channels_dict_df_p['mean_E'] = cdf.varget("P_energy")

        metadata.update({'channels_dict_df_He': channels_dict_df_He})
        metadata.update({'channels_dict_df_p': channels_dict_df_p})
    return metadata


def soho_load(dataset, startdate, enddate, path=None, resample=None, pos_timestamp=None, max_conn=5):
    """
    Downloads CDF files via SunPy/Fido from CDAWeb for CELIAS, EPHIN, ERNE onboard SOHO

    Parameters
    ----------
    dataset : {str}
        Name of SOHO dataset:
        - 'SOHO_COSTEP-EPHIN_L2-1MIN': SOHO COSTEP-EPHIN Level2 1 minute data
            https://www.ieap.uni-kiel.de/et/ag-heber/costep/data.php
            http://ulysses.physik.uni-kiel.de/costep/level2/rl2/
        - 'SOHO_COSTEP-EPHIN_L3I-1MIN': SOHO COSTEP-EPHIN Level3 intensity 1 minute data
            https://cdaweb.gsfc.nasa.gov/misc/NotesS.html#SOHO_COSTEP-EPHIN_L3I-1MIN
        - 'SOHO_CELIAS-PM_30S': SOHO CELIAS-PM 30 second data
            https://cdaweb.gsfc.nasa.gov/misc/NotesS.html#SOHO_CELIAS-PM_30S
        - 'SOHO_CELIAS-SEM_15S': SOHO CELIAS-SEM 15 second data
            https://cdaweb.gsfc.nasa.gov/misc/NotesS.html#SOHO_CELIAS-SEM_15S
        - 'SOHO_ERNE-LED_L2-1MIN': SOHO ERNE-LED Level2 1 minute data - VERY OFTEN NO DATA!
            https://cdaweb.gsfc.nasa.gov/misc/NotesS.html#SOHO_ERNE-LED_L2-1MIN
        - 'SOHO_ERNE-HED_L2-1MIN': SOHO ERNE-HED Level2 1 minute data
            https://cdaweb.gsfc.nasa.gov/misc/NotesS.html#SOHO_ERNE-HED_L2-1MIN
    startdate, enddate : {datetime or str}
        Datetime object (e.g., dt.date(2021,12,31) or dt.datetime(2021,4,15)) or "standard"
        datetime string (e.g., "2021/04/15") (enddate must always be later than startdate)
    path : {str}, optional
        Local path for storing downloaded data, by default None
    resample : {str}, optional
        Resample frequency in format understandable by Pandas, e.g. '1min', by default None
    pos_timestamp : {str}, optional
        Change the position of the timestamp: 'center' or 'start' of the accumulation interval, by default None
    max_conn : {int}, optional
        The number of parallel download slots used by Fido.fetch, by default 5

    Returns
    -------
    df : {Pandas dataframe}
        See links above for the different datasets for a description of the dataframe columns
    metadata : {dict}
        Dictionary containing different metadata, e.g., energy channels
    """
    if not (pos_timestamp=='center' or pos_timestamp=='start' or pos_timestamp is None):
        raise ValueError(f'"pos_timestamp" must be either None, "center", or "start"!')

    if dataset == 'SOHO_COSTEP-EPHIN_L2-1MIN':
        df, metadata = soho_ephin_loader(startdate, enddate, resample=resample, path=path, all_columns=False, pos_timestamp=pos_timestamp)
    else:
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

            # downloaded_files = Fido.fetch(result, path=path, max_conn=max_conn)  # use Fido.fetch(result, path='/ThisIs/MyPath/to/Data/{file}') to use a specific local folder for saving data files
            # downloaded_files.sort()
            data = TimeSeries(downloaded_files, concatenate=True)
            df = data.to_dataframe()

            metadata = _get_metadata(dataset, downloaded_files[0])

            # remove this (i.e. following lines) when sunpy's read_cdf is updated,
            # and FILLVAL will be replaced directly, see
            # https://github.com/sunpy/sunpy/issues/5908
            df = df.replace(-1e+31, np.nan)  # for all fluxes
            df = df.replace(-2147483648, np.nan)  # for ERNE count rates

            # careful!
            # adjusting the position of the timestamp manually.
            # requires knowledge of the original time resolution and timestamp position!
            if pos_timestamp == 'center':
                if (dataset.upper() == 'SOHO_ERNE-HED_L2-1MIN' or
                        dataset.upper() == 'SOHO_ERNE-LED_L2-1MIN' or
                        dataset.upper() == 'SOHO_COSTEP-EPHIN_L3I-1MIN'):
                    df.index = df.index+pd.Timedelta('30s')
                if dataset.upper() == 'SOHO_CELIAS-PM_30S':
                    df.index = df.index+pd.Timedelta('15s')
            if pos_timestamp == 'start':
                if dataset.upper() == 'SOHO_CELIAS-SEM_15S':
                    df.index = df.index-pd.Timedelta('7.5s')

            if isinstance(resample, str):
                df = resample_df(df, resample, pos_timestamp=pos_timestamp)
        except RuntimeError:
            print(f'Unable to obtain "{dataset}" data!')
            downloaded_files = []
            df = []
            metadata = []
    return df, metadata


def calc_av_en_flux_ERNE(df, channels_dict_df, avg_channels, species='p', sensor='HET'):
    """
    avg_channels : list of int, optional
        averaging channels m to n if [m, n] is provided (both integers), by default None
    """
    # calculation of total delta-E for averaging multiple channels:
    if len(avg_channels) > 1:
        DE_total = channels_dict_df.loc[avg_channels[0]:avg_channels[-1]]['DE'].sum()
    else:
        DE_total = channels_dict_df.loc[avg_channels[0]]['DE']

    # averaging of intensities:
    t_flux = 0
    for bins in range(avg_channels[0], avg_channels[-1]+1):
        if species.lower() in ['he', 'a', 'alpha']:
            t_flux = t_flux + df[f'A{sensor.upper()[0]}_{bins}'] * channels_dict_df.loc[bins]['DE']
        elif species.lower() in ['p', 'i', 'h']:
            t_flux = t_flux + df[f'P{sensor.upper()[0]}_{bins}'] * channels_dict_df.loc[bins]['DE']
    avg_flux = t_flux/DE_total

    # string lower energy
    energy_low = channels_dict_df.lower_E[avg_channels[0]]

    # string upper energy without .0 decimal but with ' keV' ending
    energy_up = channels_dict_df.upper_E[avg_channels[-1]]

    new_ch_string = f'{energy_low} - {energy_up} MeV'

    return avg_flux, new_ch_string


def soho_ephin_download(date, path=None):
    """Download SOHO/EPHIN level 2 data file from Kiel university to local path

    Parameters
    ----------
    date : datetime object
        datetime of data to retrieve
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

    doy = int(date.strftime('%j'))
    year = date.year
    if year<2000:
        pre="eph"
        yy=year-1900
    else:
        pre="epi"
        yy=year-2000
    name="%s%02d%03d" %(pre, yy, doy)
    base = "http://ulysses.physik.uni-kiel.de/costep/level2/rl2/"
    file = name+".rl2"
    url = base+str(date.year)+'/'+file

    try:
        downloaded_file = pooch.retrieve(url=url, known_hash=None, fname=file, path=path, progressbar=True)
    except ModuleNotFoundError:
        downloaded_file = pooch.retrieve(url=url, known_hash=None, fname=file, path=path, progressbar=False)
    except requests.HTTPError:
        print(f'No corresponding EPHIN data found at {url}')
        downloaded_file = []

    return downloaded_file


def soho_ephin_loader(startdate, enddate, resample=None, path=None, all_columns=False, pos_timestamp=None, use_uncorrected_data_on_own_risk=False):
    """Loads SOHO/EPHIN data and returns it as Pandas dataframe together with a dictionary providing the energy ranges per channel

    Parameters
    ----------
    startdate : str
        start date
    enddate : str
        end date
    resample : str, optional
        resample frequency in format understandable by Pandas, e.g. '1min', by default None
    path : str, optional
        local path where the files are/should be stored, by default None
    all_columns : boolean, optional
        if True provide all availalbe columns in returned dataframe, by default False

    Returns
    -------
    df : Pandas dataframe
        dataframe with either 15 channels of electron or 30 channels of proton/ion fluxes and their respective uncertainties
    channels_dict_df : dict
        Pandas dataframe giving details on the measurement channels
    """

    if not path:
        path = sunpy.config.get('downloads', 'download_dir') + os.sep

    # create list of files to load:
    dates = pd.date_range(start=startdate, end=enddate, freq='D')
    filelist = []
    for i, doy in enumerate(dates.day_of_year):
        if dates[i].year<2000:
            pre = "eph"
            yy = dates[i].year-1900
        else:
            pre = "epi"
            yy = dates[i].year-2000
        name = "%s%02d%03d" %(pre, yy, doy)
        try:
            file = glob.glob(f"{path}{os.sep}{name}.rl2")[0]
        except IndexError:
            print(f"File {name}.rl2 not found locally at {path}.")
            file = soho_ephin_download(dates[i], path)
        if len(file) > 0:
            filelist.append(file)
    if len(filelist) > 0:
        filelist = np.sort(filelist)

        col_names = ['Year', 'DOY', 'MS', 'S/C Epoch', 'Status Word part 1', 'Status Word part 2',
                     'E150', 'E300', 'E1300', 'E3000', 'P4', 'P8', 'P25', 'P41',
                     'H4', 'H8', 'H25', 'H41', 'INT',
                     'P4 GM', 'P4 GR', 'P4 S', 'P8 GM', 'P8 GR', 'P8 S',
                     'P25 GM', 'P25 GR', 'P25 S', 'P41 GM', 'P41 GR', 'P41 S',
                     'H4 GM', 'H4 GR', 'H4 S1', 'H4 S23', 'H8 GM', 'H8 GR', 'H8 S1', 'H8 S23',
                     'H25 GM', 'H25 GR', 'H25 S1', 'H25 S23', 'H41 GM', 'H41 GR', 'H41 S1', 'H41 S23',
                     'Status Flag', 'Spare 1', 'Spare 2', 'Spare 3']

        # read files into Pandas dataframes:
        df = pd.read_csv(filelist[0], header=None, sep=r'\s+', names=col_names)
        if len(filelist) > 1:
            for file in filelist[1:]:
                t_df = pd.read_csv(file, header=None, sep=r'\s+', names=col_names)
                df = pd.concat([df, t_df])

        # # generate datetime index from year, day of year, and milisec of day:
        df.index = doy2dt(df.Year.values, df.DOY.values + df.MS.values/1000./86400.)
        df.index.name = 'time'

        # drop some unused columns:
        if not all_columns:
            df = df.drop(columns=['Year', 'DOY', 'MS', 'S/C Epoch',
                                  'Status Word part 1', 'Status Word part 2',
                                  'P4 GM', 'P4 GR', 'P4 S',
                                  'P8 GM', 'P8 GR', 'P8 S',
                                  'P25 GM', 'P25 GR', 'P25 S',
                                  'P41 GM', 'P41 GR', 'P41 S',
                                  'H4 GM', 'H4 GR', 'H4 S1', 'H4 S23',
                                  'H8 GM', 'H8 GR', 'H8 S1', 'H8 S23',
                                  'H25 GM', 'H25 GR', 'H25 S1', 'H25 S23',
                                  'H41 GM', 'H41 GR', 'H41 S1', 'H41 S23',
                                  'Spare 1', 'Spare 2', 'Spare 3'])

        # Proton and helium measurements need to be corrected for effects determined post-launch,
        # cf. chapter 2.3 of https://www.ieap.uni-kiel.de/et/ag-heber/costep/materials/L2_spec_ephin.pdf
        # Until this correction has been implemented here, these data products are set to -9e9.
        # Setting use_uncorrected_data_on_own_risk=True skips this replacement, so that the uncorrected
        # data can be obtained at own risk!
        if use_uncorrected_data_on_own_risk:
            warnings.warn("Proton and helium data is still uncorrected! Know what you're doing and use at own risk!")
        else:
            df.P4 = -9e9
            df.P8 = -9e9
            df.P25 = -9e9
            df.P41 = -9e9
            df.H4 = -9e9
            df.H8 = -9e9
            df.H25 = -9e9
            df.H41 = -9e9

        # replace bad data with np.nan:
        # there shouldn't be bad data in rl2 files!
        # df = df.replace(-9999.900, np.nan)

        # derive instrument status and dependencies
        status = df['Status Flag'].values

        fmodes = np.zeros(len(status))
        for q in range(len(status)):
            binaries = '{0:08b}'.format(int(status[q]))
            if int(binaries[-1]) == 1:
                if int(binaries[-3]) == 1:
                    fmodes[q] = 1
                else:
                    fmodes[q] = 2

        ringoff = np.zeros(len(status))
        for q in range(len(status)):
            binaries = '{0:08b}'.format(int(status[q]))
            if int(binaries[-2]):
                ringoff[q] = 1

        cs_e300 = '0.67 - 3.0 MeV'
        cs_e1300 = '2.64 - 6.18 MeV'
        cs_p25 = '25 - 41 MeV'
        cs_he25 = '25 - 41 MeV/N'
        if max(fmodes)==1:
            cs_e1300 = "2.64 - 10.4 MeV"
            cs_p25 = '25 - 53 MeV'
            cs_he25 = '25 - 53 MeV/n'
        if max(fmodes)==2:
            warnings.warn('Careful: EPHIN ring off!')

        # failure mode D since 4 Oct 2017:
        # dates[-1].date() is enddate, used to catch cases when enddate is a string
        if dates[-1].date() >= dt.date(2017, 10, 4):
            cs_e300 = 'deactivated bc. of failure mode D'
            cs_e1300 = "0.67 - 10.4 MeV"
            # dates[0].date() is startdate, used to catch cases when startdate is a string
            if dates[0].date() <= dt.date(2017, 10, 4):
                warnings.warn('EPHIN instrument status (i.e., electron energy channels) changed during selected period (on Oct 4, 2017)!')

        # careful!
        # adjusting the position of the timestamp manually.
        # requires knowledge of the original time resolution and timestamp position!
        if pos_timestamp == 'center':
            df.index = df.index+pd.Timedelta('30s')

        # optional resampling:
        if isinstance(resample, str):
            df = resample_df(df, resample, pos_timestamp=pos_timestamp)
    else:
        df = []

    meta = {'E150': '0.25 - 0.7 MeV',
            'E300': cs_e300,
            'E1300': cs_e1300,
            'E3000': '4.80 - 10.4 MeV',
            'P4': '4.3 - 7.8 MeV',
            'P8': '7.8 - 25 MeV',
            'P25': cs_p25,
            'P41': '41 - 53 MeV',
            'H4': '4.3 - 7.8 MeV/n',
            'H8': '7.8 - 25.0 MeV/n',
            'H25': cs_he25,
            'H41': '40.9 - 53.0 MeV/n',
            'INT': '>25 MeV integral'}

    return df, meta


def doy2dt(year, doy):
    """
    convert decimal day of year to datetime
    """

    if isinstance(year, (int, np.uint)) or isinstance(year, (float)):
        year = [year]
    if isinstance(doy, (int, np.uint)) or isinstance(doy, (float)):
        doy = [doy]
    if len(doy) > len(year):
        year = np.zeros(len(doy))+year
    datearray = []
    for i in range(len(doy)):
        if np.isnan(doy[i]):
            datearray.append(np.nan)
        else:
            datearray.append((dt.datetime(int(year[i]), 1, 1, 0) + dt.timedelta(float(doy[i])-1)))
    return np.array(datearray)


# Followig function replaced by soho_ephin_loader()
# def ephin_rl2_downloader(startdate, enddate, path=None):
#     """
#     download EPHIN level2 rl2 files with sunpy.fido from VSO (http://virtualsolar.org)
#     """
#     # add a OS-specific '/' to end end of 'path'
#     if path:
#         if not path[-1] == os.sep:
#             path = f'{path}{os.sep}'

#     trange = a.Time(startdate, enddate)
#     result = Fido.search(trange, a.Instrument.costep)

#     if result.file_num == 0:
#         warnings.warn('WARNING: No corresponding data files found at VSO!')
#         downloaded_files = []
#     else:
#         # get list of file types of results:
#         ftypes = np.array([result.show('fileid')[0][i][0].split('.')[-1] for i in range(result.file_num)])

#         # download only .rl2 files:
#         downloaded_files = Fido.fetch(result[0][np.where(ftypes=='rl2')], path=path)

#     return downloaded_files
