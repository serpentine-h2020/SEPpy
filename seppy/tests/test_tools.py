
from astropy.utils.data import get_pkg_data_filename
from pathlib import Path
from seppy.tools import Event
import datetime
import os
import pandas as pd


def test_onset_ERNE_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 29)
    lpath = f"{os.getcwd()}/data/"
    Event1 = Event(spacecraft='SOHO', sensor='ERNE-HED', data_level='l2', species='protons', start_date=startdate, end_date=enddate, data_path=lpath)
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='sun', background_range=background_range, channels=3, resample_period="5min", yscale='log', cusum_window=30)

    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2021-10-28 16:53:05.357000')
    assert onset_found
    assert peak_time == pd.Timestamp('2021-10-28 22:53:05.357000')
    assert fig.get_axes()[0].get_title() == 'SOHO/ERNE 25.0 - 32.0 MeV protons\n5min averaging'


def test_onset_ERNE_offline():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 29)
    fullpath = get_pkg_data_filename('data/test/soho_erne-hed_l2-1min_20211028_v01.cdf', package='seppy')
    lpath = Path(fullpath).parent.as_posix()
    Event1 = Event(spacecraft='SOHO', sensor='ERNE-HED', data_level='l2', species='protons', start_date=startdate, end_date=enddate, data_path=lpath)
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='sun', background_range=background_range, channels=3, resample_period="5min", yscale='log', cusum_window=30)

    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2021-10-28 16:53:05.357000')
    assert onset_found
    assert peak_time == pd.Timestamp('2021-10-28 22:53:05.357000')
    assert fig.get_axes()[0].get_title() == 'SOHO/ERNE 25.0 - 32.0 MeV protons\n5min averaging'


def test_dynamic_spectrum_ERNE_offline():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 29)
    fullpath = get_pkg_data_filename('data/test/soho_erne-hed_l2-1min_20211028_v01.cdf', package='seppy')
    lpath = Path(fullpath).parent.as_posix()
    radio_spacecraft = None  # use ('ahead', 'STEREO-A') if #27 is fixed and radio files can be provided offline
    Event1 = Event(spacecraft='SOHO', sensor='ERNE-HED', data_level='l2', species='protons', start_date=startdate, end_date=enddate, data_path=lpath, radio_spacecraft=radio_spacecraft)
    Event1.dynamic_spectrum(view=None)

    assert Event1.fig.get_axes()[0].get_title() == 'SOHO/ERNE protons, 2021-10-28'
