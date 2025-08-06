import glob
import os
import pooch
import requests
import sunpy
import warnings
import numpy as np
import pandas as pd

from seppy.util import resample_df

# omit Pandas' PerformanceWarning
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


logger = pooch.get_logger()
logger.setLevel("WARNING")


def bepi_sixsp_download(date, path=None):
    """Download BepiColombo/SIXS-P level 3 data file from SERPENTINE data server to local path

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

    base = "https://data.serpentine-h2020.eu/l3data/bepi/l3/six_der"

    fname = f"six_der_sc_{date.year}{date.strftime('%m')}_l3_data.csv"

    url = f"{base}/{date.year}/{fname}"

    try:
        downloaded_file = pooch.retrieve(url=url, known_hash=None, fname=fname, path=path, progressbar=True)
    except ModuleNotFoundError:
        downloaded_file = pooch.retrieve(url=url, known_hash=None, fname=fname, path=path, progressbar=False)
    except requests.HTTPError:
        print(f'No corresponding BepiColombo/SIXS-P data found at {url}')
        downloaded_file = []
    print('')

    return downloaded_file


def bepi_sixsp_loader(startdate, enddate, resample=None, path=None, pos_timestamp='center'):
    """Loads BepiColombo/SIXS-P level 3 data and returns it as Pandas dataframe together with a dictionary providing the energy ranges per channel

    Parameters
    ----------
    startdate : str or datetime-like
        start date
    enddate : str or datetime-like
        end date
    resample : str, optional
        resample frequency in format understandable by Pandas, e.g. '1min', by default None
    path : str, optional
        local path where the files are/should be stored, by default None, in which case the sunpy download folder will be used.
    pos_timestamp : str, optional
        change the position of the timestamp: 'center' or 'start' of the accumulation interval, by default 'center'.

    Returns
    -------
    df : Pandas dataframe
        dataframe with either 15 channels of electron or 30 channels of proton/ion fluxes and their respective uncertainties
    channels_dict_df : dict
        Pandas dataframe giving details on the measurement channels
    """



    # <--------------------
    # channel dicts:
    ch_strings = ['45.0-55.0 keV', '55.0-65.0 keV', '65.0-75.0 keV', '75.0-85.0 keV', '85.0-105.0 keV', '105.0-125.0 keV', '125.0-145.0 keV', '145.0-165.0 keV', '165.0-195.0 keV', '195.0-225.0 keV', '225.0-255.0 keV', '255.0-295.0 keV', '295.0-335.0 keV', '335.0-375.0 keV', '375.0-425.0 keV']
    mean_E = []
    for i in range(len(ch_strings)):
        temp  = ch_strings[i].split(' keV')
        clims = temp[0].split('-')
        lower = float(clims[0])
        upper = float(clims[1])
        mean_E.append(np.sqrt(upper*lower))

    echannels = {'bins': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                 'ch_strings': ch_strings,
                 'DE': [0.0100, 0.0100, 0.0100, 0.0100, 0.0200, 0.0200, 0.0200, 0.0200, 0.0300, 0.0300, 0.0300, 0.0400, 0.0400, 0.0400, 0.0500],
                 'mean_E': mean_E}

    # create Pandas Dataframe from channels_dict:
    channels_dict_df = pd.DataFrame.from_dict(echannels)
    channels_dict_df.index = channels_dict_df.bins
    channels_dict_df.drop(columns=['bins'], inplace=True)
    # ----------------------->




    if not path:
        path = sunpy.config.get('downloads', 'download_dir') + os.sep
    # create list of files to load:
    dates = pd.date_range(start=startdate, end=enddate, freq='MS')
    filelist = []
    for i, doy in enumerate(dates.month):
        try:
            f = glob.glob(f"{path}{os.sep}six_der_sc_{dates[i].year}{dates[i].strftime('%m')}_l3_data.csv")[0] # sept_{dates[i].year}_{doy}_*.dat")[0]
        except IndexError:
            # print(f"File not found locally from {path}, downloading...")
            f = bepi_sixsp_download(dates[i], path)
        if len(f) > 0:
            filelist.append(f)
    if len(filelist) > 0:
        filelist = np.sort(filelist)

        # read files into Pandas dataframes:
        df = pd.read_csv(filelist[0])
        if len(filelist) > 1:
            for f in filelist[1:]:
                t_df = pd.read_csv(f)
                df = pd.concat([df, t_df])

        # generate datetime index:
        df.index = pd.to_datetime(df['TimeUTC'])
        df.index.name = 'TimeUTC'
        df.drop(['TimeUTC'], inplace=True, axis=1)

        # replace bad data with np.nan:
        # df = df.replace(-9999.900, np.nan)

        # TODO: (as it's not really nicely done so far)
        # careful!
        # adjusting the position of the timestamp manually.
        # requires knowledge of the original time resolution and timestamp position!
        if pos_timestamp == 'start':
            df.index = df.index-pd.Timedelta('60s')

        # optional resampling:
        if isinstance(resample, str):
            df = resample_df(df, resample, pos_timestamp=pos_timestamp)
    else:
        df = []

    return df, channels_dict_df
