import glob
import os
import pooch
import requests
import sunpy
import warnings
import numpy as np
import pandas as pd

from astropy.utils.data import get_pkg_data_filename
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

    return downloaded_file


def bepi_sixsp_l3_loader(startdate, enddate=None, resample=None, path=None, pos_timestamp='center'):
    """Loads BepiColombo/SIXS-P level 3 data and returns it as Pandas dataframe together with a dictionary providing the energy ranges per channel

    Parameters
    ----------
    startdate : str or datetime-like
        start date
    enddate : str or datetime-like, optional
        end date
    resample : str, optional
        resample frequency in format understandable by Pandas, e.g. '1min', by
        default None. Note that this is just a simple wrapper around thepandas
        resample function that is calculating the mean of the data in the new
        time bins. This is not necessarily the correct way to resample data,
        depending on the data type (for example for errors)!
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

    channels_dict = {}
    for side in range(0, 4):  # omit Side4 info because it's not part of the L3 data product (22 Aug 2025)
        for species in ['e', 'p', 'pe']:
            filepath = get_pkg_data_filename(f'data/bepi_sixsp_instrumental_constants/sixsp_side{side}_{species}_gf_en.csv', package='seppy')
            tdf = pd.read_csv(filepath, index_col=0).T
            if species == 'e':
                species_str = 'Electron'
                channels_dict[f'Side{side}_{species_str}_Bins_str'] = ((tdf['E']*1000).round(0).astype(int).astype('str')+' keV').to_dict()
            if species == 'p':
                species_str = 'Proton'
                channels_dict[f'Side{side}_{species_str}_Bins_str'] = (tdf['E'].round(2).astype('str')+' MeV').to_dict()
            if species == 'pe':
                species_str = 'Proton_As_Electron'
                for i in ['PE4', 'PE5', 'PE6']:
                    try:
                        tdf.drop(index=i, inplace=True)  # drop PE4, PE5, PE6 info because it's not part of the L3 data product (22 Aug 2025)
                    except KeyError:
                        pass
                channels_dict[f'Side{side}_{species_str}_Bins_str'] = (tdf['E'].round(2).astype('str')+' MeV').to_dict()
            channels_dict[f'Side{side}_{species_str}_Bins_Effective_Energy'] = tdf['E'].to_dict()
            channels_dict[f'Side{side}_{species_str}_Bins_Low_Energy'] = tdf['E_low'].to_dict()
            channels_dict[f'Side{side}_{species_str}_Bins_High_Energy'] = tdf['E_high'].to_dict()

    if not path:
        path = sunpy.config.get('downloads', 'download_dir') + os.sep
    #
    if not enddate:
        enddate = startdate
    startdate = sunpy.time.parse_time(startdate).to_datetime()
    enddate = sunpy.time.parse_time(enddate).to_datetime()
    if startdate.date() == enddate.date():
        enddate = enddate + pd.Timedelta('1D')
    # create list of files to load:
    dates = pd.date_range(start=startdate.replace(day=1), end=enddate, freq='MS')
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

        # shrink dataframe to requested time interval
        df = df[(df.index >= pd.to_datetime(startdate, utc=True)) & (df.index <=  pd.to_datetime(enddate, utc=True))]

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
            if len(df) > 0:
                df = resample_df(df, resample, pos_timestamp=pos_timestamp)
    else:
        df = []

    return df, channels_dict
