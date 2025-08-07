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


def bepi_sixsp_l3_loader(startdate, enddate, resample=None, path=None, pos_timestamp='center'):
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
        Pandas dataframe of measured fluxes and uncertaintites
    channels_dict_df : dict
        Dictionary giving details on the measurement channels
    """

    # TODO:
    # 1. verify values for Low and High energies
    # 2. update PE low and high energies with real numbers
    channels_dict = {"Side0_Energy_Bin_str": {'E1': '71 keV', 'E2': '106 keV', 'E3': '169 keV', 'E4': '278 keV', 'E5': '918 keV', 'E6': '2.12 MeV', 'E7': '7.01 MeV',
                                              'P1': '1.1 MeV', 'P2': '1.2 MeV', 'P3': '1.5 MeV', 'P4': '2.3 MeV', 'P5': '3.9 MeV', 'P6': '8.0 MeV', 'P7': '14.3 MeV', 'P8': '25.4 MeV', 'P9': '49.8 MeV',
                                              'PE1': '0.47 MeV', 'PE2': '0.82 MeV', 'PE3': '2.22 MeV'},
                    "Side0_Electron_Bins_Effective_Energy": np.array([0.0714, 0.106, 0.169, 0.278, 0.918, 2.120, 7.010]),
                    "Side0_Electron_Bins_Low_Energy": np.array([0.055, 0.078, 0.134, 0.235, 1.000, 1.432, 4.904]),
                    "Side0_Electron_Bins_High_Energy": np.array([0.092, 0.143, 0.214, 0.331, 1.193, 3.165, 10.000]),
                    "Side0_Proton_Bins_Effective_Energy": np.array([1.09, 1.19, 1.51, 2.26, 3.94, 8.02, 14.3, 25.4, 49.8]),
                    "Side0_Proton_Bins_Low_Energy": np.array([0.001, 1.088, 1.407, 2.139, 3.647, 7.533, 13.211, 22.606, 29.246]),
                    "Side0_Proton_Bins_High_Energy": np.array([1.254, 1.311, 1.608, 2.388, 4.241, 8.534, 15.515, 28.413, 40.0]),
                    "Side0_Proton_As_Electron_Bins_Effective_Energy": np.array([0.468, 0.824, 2.22]),
                    "Side0_Proton_As_Electron_Bins_Low_Energy": np.array([np.nan, np.nan, np.nan]),
                    "Side0_Proton_As_Electron_Bins_High_Energy": np.array([np.nan, np.nan, np.nan])}
    for i in range(1, 5):
        channels_dict[f"Side{i}_Energy_Bin_str"] = {'E1': '73 keV', 'E2': '107 keV', 'E3': '168 keV', 'E4': '275 keV', 'E5': '918 keV', 'E6': '2.22 MeV', 'E7': '6.46 MeV',
                                                    'P1': '1.1 MeV', 'P2': '1.2 MeV', 'P3': '1.5 MeV', 'P4': '2.3 MeV', 'P5': '3.9 MeV', 'P6': '8.0 MeV', 'P7': '14.5 MeV', 'P8': '25.1 MeV', 'P9': '49.8 MeV',
                                                    'PE1': '0.46 MeV', 'PE2': '0.86 MeV', 'PE3': '2.47 MeV'}
        channels_dict[f"Side{i}_Electron_Bins_Effective_Energy"] = np.array([0.0726, 0.107, 0.168, 0.275, 0.918, 2.220, 6.460])
        channels_dict[f"Side{i}_Electron_Bins_Low_Energy"] = np.array([0.055, 0.078, 0.134, 0.235, 1.000, 1.432, 4.904])
        channels_dict[f"Side{i}_Electron_Bins_High_Energy"] = np.array([0.092, 0.143, 0.214, 0.331, 1.193, 3.165, 10.000])
        channels_dict[f"Side{i}_Proton_Bins_Effective_Energy"] = np.array([1.12, 1.22, 1.53, 2.28, 3.94, 8.02, 14.5, 25.1, 49.8])
        channels_dict[f"Side{i}_Proton_Bins_Low_Energy"] = np.array([0.001, 1.088, 1.407, 2.139, 3.647, 7.533, 13.211, 22.606, 29.246])
        channels_dict[f"Side{i}_Proton_Bins_High_Energy"] = np.array([1.254, 1.311, 1.608, 2.388, 4.241, 8.534, 15.515, 28.413, 40.0])
        
        channels_dict[f"Side{i}_Proton_As_Electron_Bins_Effective_Energy"] = np.array([0.459, 0.854, 2.47])
        channels_dict[f"Side{i}_Proton_As_Electron_Bins_Low_Energy"] = np.array([np.nan, np.nan, np.nan])
        channels_dict[f"Side{i}_Proton_As_Electron_Bins_High_Energy"] = np.array([np.nan, np.nan, np.nan])


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

    return df, channels_dict
