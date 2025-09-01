# Licensed under a 3-clause BSD style license - see LICENSE.rst

import cdflib
import glob
import os
import pooch
import requests
import warnings
import datetime as dt
import numpy as np
import pandas as pd
import sunpy

from sunpy import config
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries

from seppy.util import resample_df

# omit Pandas' PerformanceWarning
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


logger = pooch.get_logger()
logger.setLevel("WARNING")


def stereo_sept_download(date, spacecraft, species, viewing, path=None):
    """Download STEREO/SEPT level 2 data file from Kiel university to local path

    Parameters
    ----------
    date : datetime object
        datetime of data to retrieve
    spacecraft : str
        'ahead' or 'behind'
    species : str
        'ele' or 'ion'
    viewing : str
        'sun', 'asun', 'north', 'south' - viewing direction of instrument
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

    if species.lower() == 'e':
        species = 'ele'
    if species.lower() == 'p' or species.lower() == 'h' or species.lower() == 'i':
        species = 'ion'

    if spacecraft.lower() == 'ahead' or spacecraft.lower() == 'a':
        base = "http://www2.physik.uni-kiel.de/STEREO/data/sept/level2/ahead/1min/"
    elif spacecraft.lower() == 'behind' or spacecraft.lower() == 'b':
        base = "http://www2.physik.uni-kiel.de/STEREO/data/sept/level2/behind/1min/"

    file = "sept_"+spacecraft.lower()+"_"+species.lower()+"_"+viewing.lower()+"_"+str(date.year)+"_"+date.strftime('%j')+"_1min_l2_v03.dat"

    url = base+str(date.year)+'/'+file

    try:
        downloaded_file = pooch.retrieve(url=url, known_hash=None, fname=file, path=path, progressbar=True)
    except ModuleNotFoundError:
        downloaded_file = pooch.retrieve(url=url, known_hash=None, fname=file, path=path, progressbar=False)
    except requests.HTTPError:
        print(f'No corresponding SEPT data found at {url}')
        downloaded_file = []
    print('')

    return downloaded_file


def stereo_sept_loader(startdate, enddate, spacecraft, species, viewing, resample=None, path=None, all_columns=False, pos_timestamp='center'):
    """Loads STEREO/SEPT data and returns it as Pandas dataframe together with a dictionary providing the energy ranges per channel

    Parameters
    ----------
    startdate : str
        start date
    enddate : str
        end date
    spacecraft : str
        STEREO spacecraft 'a'head or 'b'ehind
    species : str
        particle species: 'e'lectrons or 'p'rotons (resp. ions)
    viewing : str
        'sun', 'asun', 'north', 'south' - viewing direction of instrument
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

    # catch variation of input parameters:
    if species.lower() == 'e':
        species = 'ele'
    if species.lower() == 'p' or species.lower() == 'h' or species.lower() == 'i':
        species = 'ion'
    if spacecraft.lower() == 'a' or spacecraft.lower() == 'sta':
        spacecraft = 'ahead'
    if spacecraft.lower() == 'b' or spacecraft.lower() == 'stb':
        spacecraft = 'behind'

    # channel dicts from Nina:
    ch_strings = ['45.0-55.0 keV', '55.0-65.0 keV', '65.0-75.0 keV', '75.0-85.0 keV', '85.0-105.0 keV', '105.0-125.0 keV', '125.0-145.0 keV', '145.0-165.0 keV', '165.0-195.0 keV', '195.0-225.0 keV', '225.0-255.0 keV', '255.0-295.0 keV', '295.0-335.0 keV', '335.0-375.0 keV', '375.0-425.0 keV']
    mean_E = []
    for i in range(len(ch_strings)):
        temp  = ch_strings[i].split(' keV')
        clims = temp[0].split('-')
        lower = float(clims[0])
        upper = float(clims[1])
        mean_E.append(np.sqrt(upper*lower))
    #
    echannels = {'bins': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                 'ch_strings': ch_strings,
                 'DE': [0.0100, 0.0100, 0.0100, 0.0100, 0.0200, 0.0200, 0.0200, 0.0200, 0.0300, 0.0300, 0.0300, 0.0400, 0.0400, 0.0400, 0.0500],
                 'mean_E': mean_E}
    pchannels = {'bins': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                 'ch_strings': ['84.1-92.7 keV', '92.7-101.3 keV', '101.3-110.0 keV', '110.0-118.6 keV', '118.6-137.0 keV', '137.0-155.8 keV', '155.8-174.6 keV', '174.6-192.6 keV', '192.6-219.5 keV', '219.5-246.4 keV', '246.4-273.4 keV', ' 273.4-312.0 keV', '312.0-350.7 keV', '350.7-389.5 keV', '389.5-438.1 keV', '438.1-496.4 keV', '496.4-554.8 keV', ' 554.8-622.9 keV', '622.9-700.7 keV', '700.7-788.3 keV', '788.3-875.8 keV', '875.8- 982.8 keV', '982.8-1111.9 keV', '1111.9-1250.8 keV', '1250.8-1399.7 keV', '1399.7-1578.4 keV', '1578.4-1767.0 keV', '1767.0-1985.3 keV', '1985.3-2223.6 keV', '2223.6-6500.0 keV'],
                 'DE': [0.0086, 0.0086, 0.0087, 0.0086, 0.0184, 0.0188, 0.0188, 0.018, 0.0269, 0.0269, 0.027, 0.0386, 0.0387, 0.0388, 0.0486, 0.0583, 0.0584, 0.0681, 0.0778, 0.0876, 0.0875, 0.107, 0.1291, 0.1389, 0.1489, 0.1787, 0.1886, 0.2183, 0.2383, 4.2764],
                 'mean_E': [88.30, 96.90, 105.56, 114.22, 127.47, 146.10, 164.93, 183.38, 205.61, 232.56, 259.55, 292.06, 330.78, 369.59, 413.09, 466.34, 524.79, 587.86, 660.66, 743.21, 830.90, 927.76, 1045.36, 1179.31, 1323.16, 1486.37, 1670.04, 1872.97, 2101.07, 3801.76]}
    # :channel dicts from Nina

    if species == 'ele':
        channels_dict = echannels
    elif species == 'ion':
        channels_dict = pchannels

    # create Pandas Dataframe from channels_dict:
    channels_dict_df = pd.DataFrame.from_dict(channels_dict)
    channels_dict_df.index = channels_dict_df.bins
    channels_dict_df.drop(columns=['bins'], inplace=True)

    # column names in data files:
    # col_names = ['julian_date', 'year', 'frac_doy', 'hour', 'min', 'sec'] + \
    #             [f'ch_{i}' for i in range(2, len(channels_dict['bins'])+2)] + \
    #             [f'err_ch_{i}' for i in range(2, len(channels_dict['bins'])+2)] + \
    #             ['integration_time']
    col_names = ['julian_date', 'year', 'frac_doy', 'hour', 'min', 'sec'] + \
                [f'ch_{i}' for i in channels_dict_df.index] + \
                [f'err_ch_{i}' for i in channels_dict_df.index] + \
                ['integration_time']

    if not path:
        path = sunpy.config.get('downloads', 'download_dir') + os.sep
    # create list of files to load:
    dates = pd.date_range(start=startdate, end=enddate, freq='D')
    filelist = []
    for i, doy in enumerate(dates.day_of_year):
        try:
            file = glob.glob(f"{path}{os.sep}sept_{spacecraft}_{species}_{viewing}_{dates[i].year}_{doy}_*.dat")[0]
        except IndexError:
            # print(f"File not found locally from {path}, downloading from http://www2.physik.uni-kiel.de/STEREO/data/sept/level2/")
            file = stereo_sept_download(dates[i], spacecraft, species, viewing, path)
        if len(file) > 0:
            filelist.append(file)
    if len(filelist) > 0:
        filelist = np.sort(filelist)

        # read files into Pandas dataframes:
        df = pd.read_csv(filelist[0], header=None, sep=r'\s+', names=col_names, comment='#')
        if len(filelist) > 1:
            for file in filelist[1:]:
                t_df = pd.read_csv(file, header=None, sep=r'\s+', names=col_names, comment='#')
                df = pd.concat([df, t_df])

        # generate datetime index from Julian date:
        df.index = pd.to_datetime(df['julian_date'], origin='julian', unit='D')
        df.index.name = 'time'

        # drop some unused columns:
        if not all_columns:
            df = df.drop(columns=['julian_date', 'year', 'frac_doy', 'hour', 'min', 'sec', 'integration_time'])

        # replace bad data with np.nan:
        df = df.replace(-9999.900, np.nan)

        # TODO: (as it's not really nicely done so far)
        # careful!
        # adjusting the position of the timestamp manually.
        # requires knowledge of the original time resolution and timestamp position!
        if pos_timestamp == 'start':
            df.index = df.index-pd.Timedelta('30s')

        # optional resampling:
        if isinstance(resample, str):
            df = resample_df(df, resample, pos_timestamp=pos_timestamp)
    else:
        df = []

    return df, channels_dict_df


# def _download_metafile(dataset, path=None):
#     """
#     Download master cdf file from cdaweb for 'dataset'
#     """
#     if not path:
#         path = config.get('downloads', 'sample_dir')
#     base_url = 'https://spdf.gsfc.nasa.gov/pub/software/cdawlib/0MASTERS/'
#     fname = dataset.lower() + '_00000000_v01.cdf'
#     url = base_url + fname
#     try:
#         downloaded_file = pooch.retrieve(url=url, known_hash=None, fname=fname, path=path, progressbar=True)
#     except ModuleNotFoundError:
#         downloaded_file = pooch.retrieve(url=url, known_hash=None, fname=fname, path=path, progressbar=False)
#     return downloaded_file


# def _get_metadata(dataset, path=None):
#     """
#     Get meta data from master cdf file from cdaweb for 'dataset'
#     So far only manually for STEREO/HET
#     """
#     metadata = []
#     try:
#         if not path:
#             path = config.get('downloads', 'sample_dir')
#         if not os.path.exists(path + os.sep + dataset.lower() + '_00000000_v01.cdf'):
#             try:
#                 f = _download_metafile(dataset, path)
#             except ConnectionError:
#                 print('Found neither metadata file nor internet connection!')
#         cdf = cdflib.CDF(path + os.sep + dataset.lower() + '_00000000_v01.cdf')
#         if dataset[-3:].upper()=='HET':
#             e_mean_energies = cdf.varget('Electron_Flux_Energy_vals')
#             e_energy_bins = cdf.varget('Electron_Flux_Energies')
#             p_mean_energies = cdf.varget('Proton_Flux_Energy_vals')
#             p_energy_bins = cdf.varget('Proton_Flux_Energies')
#             metadata = {'e_mean_energies': cdf.varget('Electron_Flux_Energy_vals'),
#                         'e_energy_bins': cdf.varget('Electron_Flux_Energies'),
#                         'p_mean_energies': cdf.varget('Proton_Flux_Energy_vals'),
#                         'p_energy_bins': cdf.varget('Proton_Flux_Energies')
#                         }
#     except AttributeError:
#         metadata = []
#     except ValueError:
#         metadata = []
#     return metadata


def _get_metadata(dataset, path_to_cdf):
    """
    Get meta data from single cdf file
    So far only manually for STEREO/HET
    """
    metadata = []
    cdf = cdflib.CDF(path_to_cdf)
    if dataset[-3:].upper()=='HET':
        metadata = {'Electron_Bins_Text': cdf.varget('Electron_Flux_Energies'),
                    'Electron_Flux_UNITS': cdf.varattsget('Electron_Flux')['UNITS'],
                    'Electron_Flux_FILLVAL': cdf.varattsget('Electron_Flux')['FILLVAL'],
                    'Proton_Bins_Text': cdf.varget('Proton_Flux_Energies'),
                    'Proton_Flux_UNITS': cdf.varattsget('Proton_Flux')['UNITS'],
                    'Proton_Flux_FILLVAL': cdf.varattsget('Proton_Flux')['FILLVAL'],
                    }

        channels_dict_df_e = pd.DataFrame(metadata['Electron_Bins_Text'], columns=['ch_strings'])
        channels_dict_df_e['lower_E'] = channels_dict_df_e.ch_strings.apply(lambda x: float((x.split('-')[0]).replace(' ', '').replace('MeV', '')))
        channels_dict_df_e['upper_E'] = channels_dict_df_e.ch_strings.apply(lambda x: float((x.split('-')[1]).replace(' ', '').replace('MeV', '')))
        channels_dict_df_e['DE'] = channels_dict_df_e['upper_E'] - channels_dict_df_e['lower_E']
        channels_dict_df_e['mean_E'] = np.sqrt(channels_dict_df_e['upper_E'] * channels_dict_df_e['lower_E'])

        channels_dict_df_p = pd.DataFrame(metadata['Proton_Bins_Text'], columns=['ch_strings'])
        channels_dict_df_p['lower_E'] = channels_dict_df_p.ch_strings.apply(lambda x: float((x.split('-')[0]).replace(' ', '').replace('MeV', '')))
        channels_dict_df_p['upper_E'] = channels_dict_df_p.ch_strings.apply(lambda x: float((x.split('-')[1]).replace(' ', '').replace('MeV', '')))
        channels_dict_df_p['DE'] = channels_dict_df_p['upper_E'] - channels_dict_df_p['lower_E']
        channels_dict_df_p['mean_E'] = np.sqrt(channels_dict_df_p['upper_E'] * channels_dict_df_p['lower_E'])

        metadata.update({'channels_dict_df_e': channels_dict_df_e})
        metadata.update({'channels_dict_df_p': channels_dict_df_p})
    if dataset[-3:].upper()=='LET':
        # Lets hope the parameter names and values are always the same...
        # Let first only keep the proton unsectered data, which is the standard data product
        # Following the code format before, text, units and fillval are needed for the metadata
        #
        # First the  unsec data: i.e., the standard LET data product
        unsec_name = []
        for var in cdf.cdf_info().zVariables:
            if 'unsec' in var:
                unsec_name.append(var)
        metadata = {}

        for var in unsec_name:
            metadata.update({var: cdf.varattsget(var)['CATDESC']})
            metadata.update({var+'_UNITS': cdf.varattsget(var)['UNITS']})
            metadata.update({var+'_FILLVAL': cdf.varattsget(var)['FILLVAL']})
            metadata.update({var+'_Bins_Text': cdf.varget(cdf.varattsget(var)['DEPEND_1'])})

            channels_dict_df_p = pd.DataFrame(metadata[var+'_Bins_Text'], columns=['ch_strings'])
            channels_dict_df_p['lower_E'] = channels_dict_df_p.ch_strings.apply(lambda x: float((x.split('-')[0]).strip()))
            channels_dict_df_p['upper_E'] = channels_dict_df_p.ch_strings.apply(lambda x: float((x.split('-')[1]).split()[0].strip()))
            channels_dict_df_p['DE'] = channels_dict_df_p['upper_E'] - channels_dict_df_p['lower_E']
            channels_dict_df_p['mean_E'] = np.sqrt(channels_dict_df_p['upper_E'] * channels_dict_df_p['lower_E'])   
            metadata.update({'channels_dict_df_'+var: channels_dict_df_p})
        #sec data, only load proton data, others are not very commonly used
        sec_name = ['H_Lo_sec_flux', 'H_Hi_sec_flux', 'H_VLo_sec_flux', 'H_Lo_sec_cnts', 'H_Hi_sec_cnts', 'H_VLo_sec_cnts']
        for var in sec_name:
            metadata.update({var: cdf.varattsget(var)['CATDESC']})
            metadata.update({var+'_UNITS': cdf.varattsget(var)['UNITS']})
            metadata.update({var+'_FILLVAL': cdf.varattsget(var)['FILLVAL']})
        metadata.update({'H_Lo_sec_flux_Energy_Bins': [4,6]})
        metadata.update({'H_Hi_sec_flux_Energy_Bins': [6,12]})
        metadata.update({'H_VLo_sec_flux_Energy_Bins': [1.8,3.6]})
        metadata.update({'H_Lo_sec_cnts_Energy_Bins': [4,6]})
        metadata.update({'H_Hi_sec_cnts_Energy_Bins': [6,12]})
        metadata.update({'H_VLo_sec_cnts_Energy_Bins': [1.8,3.6]})
        metadata.update({'H_sec_flux_Energy_Bins':[[1.8,3.6],[4,6],[6,12]]})
    return metadata

def let(df_let, metadata, species='p', type='unsec', count_rate=False, flux=True):
    """
    Extract LET data from dataframe returned by stereo_load,
    Here are the available species:
    species : str
        'H'/'P' for protons (default) ;
        'He' for helium ions;
        'He4' for helium-4 ions;
        'He3' for helium-3 ions;
        and more
    type : str
        'unsec' for unsectered (default);
        'sec' for sectered; proton implemented
        'summed' for summed sectered and unsectered; TODO
    count_rate : bool, optional, returen count rates instead of fluxes, by default False
    flux : bool, optional, return fluxes instead of count rates, by default True
    Returns
    -------
    df_let_species : Pandas dataframe
        dataframe with either 16 channels of proton or 16 channels of helium ion fluxes and their respective uncertainties
        or other species if added in future

    ------
    usage example:
    df_let, metadata = stereo_load('LET', '2022-01-01', '2022-01-03', spacecraft='ahead', path=None)
    df_let_p = let(df_let, metadata, species='p', type='unsec', flux=True) 
    """
    _default_species = ['H', 'He', 'He4', 'He3','C','N','O','Ne','Mg','Al','Si','S','Ar','Ca','Fe','Ni']
    _default_species_lower = [s.lower() for s in _default_species]
    if type =='unsec':
        if species.lower() == 'p' or species.lower() == 'h':
            nametag = 'H_unsec_flux' if flux else 'H_unsec_cnts'
            col_names = [nametag+'_' + str(i) for i in range(0, 12)]
        elif species.lower() in _default_species_lower:
            _name =  _default_species[_default_species_lower.index(species.lower())]
            nametag = _name+'_unsec_flux' if flux else _name+'_unsec_cnts'
            DEPEND_1 = metadata[nametag+'_Bins_Text']
            col_names = [nametag+'_' + str(i) for i in range(0, len(DEPEND_1))]
        else:
            print('these speciese are not mearured by LET, available species are:', _default_species)
        df_let_species = df_let[col_names]
        df_let_species.attrs['Bins']= metadata['channels_dict_df_'+nametag]
        return df_let_species
    if type =='sec':
        if species.lower() == 'p' or species.lower() == 'h':
            nametag = 'H_Lo_sec_flux' if flux else 'H_Lo_sec_cnts'
            col_names = [nametag+'_' + str(i) for i in range(0, 16)]
            nametag = 'H_Hi_sec_flux' if flux else 'H_Hi_sec_cnts'
            col_names += [nametag+'_' + str(i) for i in range(0, 16)]
            nametag = 'H_VLo_sec_flux' if flux else 'H_VLo_sec_cnts'
            col_names += [nametag+'_' + str(i) for i in range(0, 16)]
        df_let_species = df_let[col_names]
        df_let_species.attrs['Bins']= metadata['H_sec_flux_Energy_Bins']
        return df_let_species



def het(df_het, metadata, species='e'):
    """
    Extract HET data from dataframe returned by stereo_load,
    Here are the available species:
    species : str
        'e' for electrons (default) ;
        'p' or 'h' for protons;
    Returns
    -------
    df_het_species : Pandas dataframe
    ------
    usage example:
    df_het, metadata = stereo_load('HET', '2022-01-01', '2022-01-03', spacecraft='ahead', path=None)
    df_het_e = het(df_het, metadata, species='e')
    """
    if species.lower() == 'e':
        nametag = 'Electron' 
        col_names = []
    elif species.lower() == 'p' or species.lower() == 'h':
        nametag = 'Proton'
    else:
        print('these speciese are not mearured by HET, available species are: e, p (or h)')
    col_names = []
    for key in df_het.columns:
        if nametag in key:
            col_names.append(key)
    df_het_species = df_het[col_names]
    if species.lower() == 'e':
        df_het_species.attrs['Bins']= metadata['channels_dict_df_e']
        df_het_species.attrs['Units']= metadata['Electron_Flux_UNITS']
    if species.lower() == 'p' or species.lower() == 'h':
        df_het_species.attrs['Bins']= metadata['channels_dict_df_p']
        df_het_species.attrs['Units']= metadata['Proton_Flux_UNITS']
    return df_het_species


def stereo_load(instrument, startdate, enddate, spacecraft='ahead', mag_coord='RTN', sept_species='e', sept_viewing=None, path=None, resample=None, pos_timestamp='center', max_conn=5):
    """
    Downloads CDF files via SunPy/Fido from CDAWeb for HET, LET, MAG, and SEPT onboard STEREO

    Parameters
    ----------
    instrument : str
        Name of STEREO instrument: \n
            - 'HET': STEREO IMPACT/HET Level 1 Data \n
              https://cdaweb.gsfc.nasa.gov/misc/NotesS.html#STA_L1_HET \n
            - 'LET': STEREO IMPACT/LET Level 1 Data \n
              https://cdaweb.gsfc.nasa.gov/misc/NotesS.html#STA_L1_LET \n
            - 'MAG': STEREO IMPACT/MAG Magnetic Field Vectors (RTN or SC => mag_coord) \n
              https://cdaweb.gsfc.nasa.gov/misc/NotesS.html#STA_L1_MAG_RTN \n
              https://cdaweb.gsfc.nasa.gov/misc/NotesS.html#STA_L1_MAG_SC \n
            - 'MAGB': STEREO IMPACT/MAG Burst Mode (~0.03 sec) Magnetic Field Vectors (RTN or SC => mag_coord) \n
              https://cdaweb.gsfc.nasa.gov/misc/NotesS.html#STA_L1_MAGB_RTN \n
              https://cdaweb.gsfc.nasa.gov/misc/NotesS.html#STA_L1_MAGB_SC \n
            - 'MAGPLASMA': STEREO IMPACT/MAG Magnetic Field and PLASTIC Solar Wind Plasma Level 2 Data \n
              https://cdaweb.gsfc.nasa.gov/misc/NotesS.html#STA_L2_MAGPLASMA_1M \n
            - 'SEPT': STEREO IMPACT/SEPT Level 2 Data
    startdate, enddate : datetime or str
        Datetime object (e.g., dt.date(2021,12,31) or dt.datetime(2021,4,15)) or "standard"
        datetime string (e.g., "2021/04/15") (enddate must always be later than startdate)
    spacecraft : str, optional
        Name of STEREO spacecraft: 'ahead' or 'behind', by default 'ahead'
    mag_coord : str, optional
        Coordinate system for MAG: 'RTN' or 'SC', by default 'RTN'
    sept_species : str, optional
        Particle species for SEPT: 'e'lectrons or 'p'rotons (resp. ions), by default 'e'
    sept_viewing : str, optional
        Viewing direction for SEPT: 'sun', 'asun', 'north', or 'south', by default 'sun'
    path : str, optional
        Local path for storing downloaded data, by default None
    resample : str, optional
        resample frequency in format understandable by Pandas, e.g. '1min', by default None
    pos_timestamp : str, optional
        change the position of the timestamp: 'center' or 'start' of the accumulation interval,
        or 'original' to do nothing, by default 'center'.
    max_conn : int, optional
        The number of parallel download slots used by Fido.fetch, by default 5


    Returns
    -------
    df : {Pandas dataframe}
        See links above for the different datasets for a description of the dataframe columns
    metadata : {dict}
        Dictionary containing different metadata, e.g., energy channels
    """
    trange = a.Time(startdate, enddate)
    if trange.min==trange.max:
        print(f'"startdate" and "enddate" might need to be different!')

    # Catch old default value for pos_timestamp
    if pos_timestamp is None:
        pos_timestamp = 'center'

    if not (pos_timestamp=='center' or pos_timestamp=='start' or pos_timestamp=='original'):
        raise ValueError(f'"pos_timestamp" must be either "original", "center", or "start"!')

    # find name variations
    if spacecraft.lower()=='a' or spacecraft.lower()=='sta':
        spacecraft='ahead'
    if spacecraft.lower()=='b' or spacecraft.lower()=='stb':
        spacecraft='behind'

    if instrument.upper()=='SEPT':
        if not sept_viewing:
            raise Exception("STEREO/SEPT loading requires a defined 'sept_viewing'!")
        else:
            df, channels_dict_df = stereo_sept_loader(startdate=startdate,
                                                      enddate=enddate,
                                                      spacecraft=spacecraft,
                                                      species=sept_species,
                                                      viewing=sept_viewing,
                                                      resample=resample,
                                                      path=path,
                                                      all_columns=False,
                                                      pos_timestamp=pos_timestamp)
            return df, channels_dict_df
    else:
        # define spacecraft string
        sc = 'ST' + spacecraft.upper()[0]

        # define dataset
        if (instrument.upper()[:3]=='MAG') & (instrument.upper()!='MAGPLASMA'):
            dataset = sc + '_L1_' + instrument.upper() + '_' + mag_coord.upper()
        elif instrument.upper()=='MAGPLASMA':
            dataset = sc + '_L2_MAGPLASMA_1M'
        # elif instrument.upper()[:3]=='PLA':
        #     dataset = sc + '_L2_PLA_1DMAX_1MIN'
        else:
            dataset = sc + '_L1_' + instrument.upper()

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
            data = TimeSeries(downloaded_files, concatenate=True)
            df = data.to_dataframe()

            metadata = _get_metadata(dataset, downloaded_files[0])

            # TODO: remove this (i.e. following two lines) when sunpy's
            # read_cdf is updated, and FILLVAL will be replaced directly, see
            # https://github.com/sunpy/sunpy/issues/5908
            # if instrument.upper() == 'HET':
            #     df = df.replace(metadata['Electron_Flux_FILLVAL'], np.nan)
            # if instrument.upper() == 'LET':
            #     df = df.replace(-1e+31, np.nan)
            #     df = df.replace(-2147483648, np.nan)
            # if instrument.upper() == 'MAG':
            #     df = df.replace(-1e+31, np.nan)
            # if instrument.upper() == 'MAGPLASMA':
            #     df = df.replace(-1.0e+30, np.nan)
            # 4 Apr 2023: previous 9 lines removed because they are taken care of with sunpy
            # 4.1.0:
            # https://docs.sunpy.org/en/stable/whatsnew/changelog.html#id7
            # https://github.com/sunpy/sunpy/pull/5956

            # Because MAGPLASMA datafiles are for full calendar years, select only
            # the requested time range from it. Using trange because startdate
            # or enddate can be strings.
            if instrument.upper() == 'MAGPLASMA':
                # df = df[(df.index>=trange.min.value) & (df.index < (trange.max+pd.Timedelta('1d')).value)]
                df = df[(df.index>=trange.min.value) & (df.index < trange.max.value)]

            # TODO: (as it's not really nicely done so far)
            # careful!
            # adjusting the position of the timestamp manually.
            # requires knowledge of the original time resolution and timestamp position!
            if pos_timestamp == 'center':
                if instrument.upper() == 'HET':
                    df.index = df.index+pd.Timedelta('30s')
            if pos_timestamp == 'start':
                if instrument.upper() == 'LET':
                    df.index = df.index-pd.Timedelta('30s')
                if instrument.upper() == 'MAGPLASMA':
                    df.index = df.index-pd.Timedelta('30s')

            if isinstance(resample, str):
                df = resample_df(df, resample, pos_timestamp=pos_timestamp)
        except (RuntimeError, IndexError):
            print(f'Unable to obtain "{dataset}" data for {startdate}-{enddate}!')
            downloaded_files = []
            df = []
            metadata = []
        return df, metadata


def calc_av_en_flux_SEPT(df, channels_dict_df, avg_channels):
    """
    avg_channels : list of int, optional
        averaging channels m to n if [m, n] is provided (both integers), by default None
    """

    # # create Pandas Dataframe from channels_dict:
    # channels_dict_df = pd.DataFrame.from_dict(channels_dict)
    # channels_dict_df.index = channels_dict_df.bins
    # channels_dict_df.drop(columns=['bins'], inplace=True)

    # calculation of total delta-E for averaging multiple channels:
    if len(avg_channels) > 1:
        # DE_total = sum(channels_dict['DE'][avg_channels[0]-2:avg_channels[-1]-2+1])
        DE_total = channels_dict_df.loc[avg_channels[0]:avg_channels[-1]]['DE'].sum()
    else:
        # DE_total = channels_dict['DE'][avg_channels[0]-2]
        DE_total = channels_dict_df.loc[avg_channels[0]]['DE']

    # averaging of intensities:
    t_flux = 0
    for bins in range(avg_channels[0], avg_channels[-1]+1):
        # t_flux = t_flux + chan_data[:, bins-2]*channels_dict['DE'][bins-2]
        t_flux = t_flux + df[f'ch_{bins}'] * channels_dict_df.loc[bins]['DE']
    avg_flux = t_flux/DE_total

    # building new channel string
    # ch_string1 = channels['ch_strings'][ch[0]-2]
    # ch_string11 = str.split(ch_string1, '-')[0]
    # ch_string11 = str.split(ch_string11, '.0')[0]
    # ch_string2 = channels['ch_strings'][ch[-1]-2]
    # ch_string22 = str.split(ch_string2, '-')[1]
    # ch_string22 = str.split(ch_string22, '.0')[0]

    # ch_string =ch_string11+'-'+ch_string22+' keV'+' '+which

    # string lower energy without .0 decimal
    energy_low = channels_dict_df.loc[avg_channels[0]]['ch_strings'].split('-')[0].replace(".0", "")

    # string upper energy without .0 decimal but with ' keV' ending
    energy_up = channels_dict_df.loc[avg_channels[-1]]['ch_strings'].split('-')[-1].replace(".0", "")

    new_ch_string = energy_low + '-' + energy_up

    return avg_flux, new_ch_string


def calc_av_en_flux_ST_HET(df, channels_dict_df, avg_channels, species):
    """
    avg_channels : list of int, optional
        averaging channels m to n if [m, n] is provided (both integers), by default None
    """
    if type(avg_channels) is int:
        avg_channels = [avg_channels]
    # calculation of total delta-E for averaging multiple channels:
    if len(avg_channels) > 1:
        DE_total = channels_dict_df.loc[avg_channels[0]:avg_channels[-1]]['DE'].sum()
    else:
        DE_total = channels_dict_df.loc[avg_channels[0]]['DE']

    # averaging of intensities:
    t_flux = 0
    for bins in range(avg_channels[0], avg_channels[-1]+1):
        if species.lower() == 'e':
            t_flux = t_flux + df[f'Electron_Flux_{bins}'] * channels_dict_df.loc[bins]['DE']
        elif species.lower() == 'p' or species.lower() == 'i' or species.lower() == 'h':
            t_flux = t_flux + df[f'Proton_Flux_{bins}'] * channels_dict_df.loc[bins]['DE']
    avg_flux = t_flux/DE_total

    # string lower energy
    energy_low = channels_dict_df.lower_E[avg_channels[0]]

    # string upper energy without .0 decimal but with ' keV' ending
    energy_up = channels_dict_df.upper_E[avg_channels[-1]]

    new_ch_string = f'{energy_low} - {energy_up} MeV'

    return avg_flux, new_ch_string
