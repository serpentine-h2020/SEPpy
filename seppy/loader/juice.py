import os

import cdflib
import pandas as pd
import pooch
import requests
import sunpy
from bs4 import BeautifulSoup
from packaging.version import Version
from seppy.util import resample_df
from sunpy.timeseries import TimeSeries

logger = pooch.get_logger()
logger.setLevel("WARNING")


def juice_radem_download(date, path=None):
    """Download JUICE/RADEM cruise science data file from ESA's PSA to local path

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
    # use sunpy download directory if no path is provided
    if not path:
        path = sunpy.config.get('downloads', 'download_dir')

    # add a OS-specific '/' to end end of 'path'
    if path:
        if not path[-1] == os.sep:
            path = f'{path}{os.sep}'

    # URL of the webpage containing the downloadable files
    base_url = f"https://archives.esac.esa.int/psa/ftp/Juice/juice_radem/data_raw/cruise/sc/{date.year}{date.strftime('%m')}/"

    # Send an HTTP GET request to the webpage
    response = requests.get(base_url)

    # Check if the request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all links on the page
        links = soup.find_all('a')

        # Filter for the file link 
        fname = None
        for link in links:
            href = link.get('href')
            if href and f"radem_raw_sc_{date.year}{date.strftime('%m')}{date.strftime('%d')}__" in href and href.endswith('.cdf'):
                fname = href
                break  # Get the first found link

        if fname:
            url = base_url + fname

            try:
                downloaded_file = pooch.retrieve(url=url, known_hash=None, fname=fname, path=path, progressbar=True)
            except ModuleNotFoundError:
                downloaded_file = pooch.retrieve(url=url, known_hash=None, fname=fname, path=path, progressbar=False)
            except requests.HTTPError:
                print(f'No corresponding JUICE/RADEM data found at {url}')
                downloaded_file = []

            # # Download the file
            # file_response = requests.get(file_url)

            # # Save the file if the request was successful
            # if file_response.status_code == 200:
            #     fname = file_url.split('/')[-1]  # Extract fname from the URL
            #     with open(fname, 'wb') as f:
            #         f.write(file_response.content)
            #     print(f"Downloaded: {fname}")
            # else:
            #     print(f"Failed to download file: {file_response.status_code}")

            return downloaded_file
        else:
            print("No suitable file found online.")
            return None
    else:
        print(f"Failed to fetch the webpage: {response.status_code}")
        return None


def juice_radem_load(startdate, enddate, resample=None, path=None, pos_timestamp='center'):
    """Download & load JUICE/RADEM cruise science data and returns it as Pandas DataFrame (and metadata dictionaries).
    Note that the data is provided in counts and not converted to physical units (as of Nov 2025); also the instrument configuration changes over time.

    Parameters
    ----------
    startdate : datetime object
        start datetime of data to retrieve
    enddate : datetime object
        end datetime of data to retrieve
    resample : str
        resampling frequency (e.g. '1min', '10min', '1H', etc.). If None, no resampling is applied.
    path : str
        local path where the files are stored / will be downloaded to
    pos_timestamp : str
        position of the timestamp when resampling ('start', 'center', 'end')

    Returns
    -------
    df : Pandas DataFrame
        DataFrame containing the JUICE/RADEM data
    energies_dict : dict
        Dictionary containing the JUICE/RADEM data energy and label information
    metadata_dict : dict
        Dictionary containing the JUICE/RADEM data metadata
    """

    # Generate list of dates between startdate and enddate
    dates = pd.date_range(start=startdate, end=enddate, freq='D')

    downloaded_files = []
    for date in dates:
        fname = juice_radem_download(date, path=path)
        if fname:
            downloaded_files.append(fname)

    if not downloaded_files:
        print("No data files were downloaded.")
        return pd.DataFrame(), {}, {}

    # Load the data using SunPy TimeSeries
    data = TimeSeries(downloaded_files, concatenate=True)
    df = data.to_dataframe()

    # drop string columns
    df.drop(columns=['TIME_OBT'], inplace=True)

    # convert TIME_UTC column from string to datetime
    df['TIME_UTC'] = pd.to_datetime(df['TIME_UTC'])

    if resample:
        df = resample_df(df, resample, pos_timestamp=pos_timestamp)

    energies_dict, metadata_dict = juice_radem_load_metadata(filename=downloaded_files[0])

    return df, energies_dict, metadata_dict


def juice_radem_load_metadata(filename):
    """Load JUICE/RADEM cruise science data metadata and return it as a dictionary

    Returns
    -------
    energies_dict : dict
        Dictionary containing the JUICE/RADEM data energy and label information
    metadata_dict : dict
        Dictionary containing the JUICE/RADEM data metadata
    """

    # open cdf file with cdflib to access metadata
    cdf = cdflib.CDF(filename)

    # dict with all metadata info
    metadata_dict = {"Global_Attributes": cdf.globalattsget()}

    # dict with energy/label infos
    energies_dict = {}

    cdf_info = cdf.cdf_info()
    if hasattr(cdflib, "__version__") and Version(cdflib.__version__) >= Version("1.0.0"):
        all_var_keys = cdf_info.rVariables + cdf_info.zVariables
    else:
        all_var_keys = cdf_info['rVariables'] + cdf_info['zVariables']
    #
    for key in all_var_keys:
        metadata_dict[key] = cdf.varattsget(key)
        if cdf.varattsget(key)['VAR_TYPE'] == 'metadata':
            energies_dict[key] = cdf.varget(key)

    return energies_dict, metadata_dict
