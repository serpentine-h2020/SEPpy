
from astropy.utils.data import get_pkg_data_filename
from pathlib import Path
from seppy.tools import Event
import datetime
import os
import pandas as pd


# TODO: find smaller datasets for SOLO/STEP
# TODO: test dynamic spectrum for all dataset
# TODO: test tsa for all dataset


def test_onset_SOLO_STEP_ions_old_data_online():
    startdate = datetime.date(2020, 9, 21)
    enddate = datetime.date(2020, 9, 21)
    lpath = f"{os.getcwd()}/data/"
    background_range = (datetime.datetime(2020, 9, 21, 0, 0, 0), datetime.datetime(2020, 9, 21, 2, 0, 0))
    #
    # ions
    Event1 = Event(spacecraft='Solar Orbiter', sensor='STEP', data_level='l2', species='ions', start_date=startdate, end_date=enddate, data_path=lpath)
    # print(Event1.print_energies())  # TODO: see test_onset_SOLO_EPT_online
    # Pixel averaged
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='Pixel averaged', background_range=background_range, channels=1, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2020-09-21 17:27:32.010263')
    assert onset_found
    assert peak_time == pd.Timestamp('2020-09-21 17:57:32.010263')
    assert fig.get_axes()[0].get_title() == 'SOLO/STEP 0.0060 - 0.0091 MeV/n protons\n5min averaging, viewing: PIXEL AVERAGED'
    # Pixel 8 - check that calculation is stopped bc. this data is not implemented correctly!
    try:
        flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='Pixel 8', background_range=background_range, channels=1, resample_period="5min", yscale='log', cusum_window=30)
    except Warning:
        check = True
    assert check


def test_onset_SOLO_STEP_ions_new_data_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 28)
    lpath = f"{os.getcwd()}/data/"
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    # ions
    Event1 = Event(spacecraft='Solar Orbiter', sensor='STEP', data_level='l2', species='ions', start_date=startdate, end_date=enddate, data_path=lpath)
    # print(Event1.print_energies())  # TODO: see test_onset_SOLO_EPT_online
    # Pixel averaged
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='Pixel averaged', background_range=background_range, channels=1, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2021-10-28 16:07:30.153419')
    assert onset_found
    assert peak_time == pd.Timestamp('2021-10-28 18:57:30.153419')
    assert fig.get_axes()[0].get_title() == 'SOLO/STEP 0.0061 - 0.0091 MeV protons\n5min averaging, viewing: PIXEL AVERAGED'
    # Pixel 8
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='Pixel 8', background_range=background_range, channels=1, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2021-10-28 16:12:30.153419')
    assert onset_found
    assert peak_time == pd.Timestamp('2021-10-28 20:52:30.153419')
    assert fig.get_axes()[0].get_title() == 'SOLO/STEP 0.0061 - 0.0091 MeV protons\n5min averaging, viewing: PIXEL 8'


def test_onset_SOLO_HET_online():
    startdate = datetime.date(2022, 11, 8)
    enddate = datetime.date(2022, 11, 8)
    lpath = f"{os.getcwd()}/data/"
    background_range = (datetime.datetime(2022, 11, 8, 0, 0, 0), datetime.datetime(2022, 11, 8, 1, 0, 0))
    # viewing "sun", single channel, protons
    Event1 = Event(spacecraft='Solar Orbiter', sensor='HET', data_level='l2', species='protons', start_date=startdate, end_date=enddate, data_path=lpath)
    # print(Event1.print_energies())  # TODO: see test_onset_SOLO_EPT_online
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='sun', background_range=background_range, channels=1, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (73,)
    assert len(onset_stats) == 6
    assert isinstance(onset_stats[5], pd._libs.tslibs.nattype.NaTType)  # onset_stats[5] == pd.Timestamp('2021-10-28 15:31:59.492059')
    assert ~onset_found
    assert peak_time == pd.Timestamp('2022-11-08 17:57:54.269660')
    assert fig.get_axes()[0].get_title() == 'SOLO/HET 7.3540 - 7.8900 MeV protons\n5min averaging, viewing: SUN'
    # viewing "north", combined channel, electrons
    Event1 = Event(spacecraft='Solar Orbiter', sensor='HET', data_level='l2', species='electrons', start_date=startdate, end_date=enddate, data_path=lpath)
    # print(Event1.print_energies())  # TODO: see test_onset_SOLO_EPT_online
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='north', background_range=background_range, channels=[0, 3], resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (73,)
    assert len(onset_stats) == 6
    assert isinstance(onset_stats[5], pd._libs.tslibs.nattype.NaTType)  # onset_stats[5] == pd.Timestamp('2021-10-28 15:31:59.492059')
    assert ~onset_found
    assert peak_time == pd.Timestamp('2022-11-08 22:27:54.269660')
    assert fig.get_axes()[0].get_title() == 'SOLO/HET 0.4533 - 18.8300 MeV electrons\n5min averaging, viewing: NORTH'


def test_onset_SOLO_EPT_online():
    startdate = datetime.date(2022, 6, 6)
    enddate = datetime.date(2022, 6, 6)
    lpath = f"{os.getcwd()}/data/"
    background_range = (datetime.datetime(2022, 6, 6, 0, 0, 0), datetime.datetime(2022, 6, 6, 1, 0, 0))
    # viewing "sun", single channel, ions
    Event1 = Event(spacecraft='Solar Orbiter', sensor='EPT', data_level='l2', species='ions', start_date=startdate, end_date=enddate, data_path=lpath)
    # print(Event1.print_energies())  # TODO: Fix bug! right now viewing is not defined. if run after event.find_onset, it works!
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='sun', background_range=background_range, channels=4, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert isinstance(onset_stats[5], pd._libs.tslibs.nattype.NaTType)  # onset_stats[5] == pd.Timestamp('2021-10-28 15:31:59.492059')
    assert ~onset_found
    assert peak_time == pd.Timestamp('2022-06-06 01:02:30.902854')
    assert fig.get_axes()[0].get_title() == 'SOLO/EPT 0.0608 - 0.0678 MeV protons\n5min averaging, viewing: SUN'
    # viewing "north", combined channel, electrons
    Event1 = Event(spacecraft='Solar Orbiter', sensor='EPT', data_level='l2', species='electrons', start_date=startdate, end_date=enddate, data_path=lpath)
    # print(Event1.print_energies())  # TODO: see above
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='north', background_range=background_range, channels=[1, 4], resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert isinstance(onset_stats[5], pd._libs.tslibs.nattype.NaTType)  # onset_stats[5] == pd.Timestamp('2021-10-28 15:31:59.492059')
    assert ~onset_found
    assert peak_time == pd.Timestamp('2022-06-06 23:02:30.902854')
    assert fig.get_axes()[0].get_title() == 'SOLO/EPT 0.0334 - 0.0439 MeV electrons\n5min averaging, viewing: NORTH'


def test_onset_PSP_ISOIS_EPIHI_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 29)
    lpath = f"{os.getcwd()}/data/"
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    # viewing "A", single channel, electrons
    Event1 = Event(spacecraft='PSP', sensor='isois-epihi', data_level='l2', species='electrons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='A', background_range=background_range, channels=4, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (194,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2021-10-28 15:31:59.492059')
    assert onset_found
    assert peak_time == pd.Timestamp('2021-10-28 16:06:59.492059')
    assert fig.get_axes()[0].get_title() == 'PSP/ISOIS-EPIHI 0.8 - 1.0 MeV electrons\n5min averaging, viewing: A'
    # viewing "B", combined channel, protons
    Event1 = Event(spacecraft='PSP', sensor='isois-epihi', data_level='l2', species='protons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='B', background_range=background_range, channels=[1, 5], resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (194,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2021-10-28 16:46:59.492059')
    assert onset_found
    assert peak_time == pd.Timestamp('2021-10-28 19:56:59.492059')
    assert fig.get_axes()[0].get_title() == 'PSP/ISOIS-EPIHI 8.0 - 19.0 MeV protons\n5min averaging, viewing: B'


def test_onset_PSP_ISOIS_EPILO_e_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 29)
    lpath = f"{os.getcwd()}/data/"
    Event1 = Event(spacecraft='PSP', sensor='isois-epilo', data_level='l2', species='electrons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    # viewing "7", single channel
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='7', background_range=background_range, channels=4, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (198,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2021-10-28 15:33:14.991967')
    assert onset_found
    assert peak_time == pd.Timestamp('2021-10-28 16:43:14.991967')
    assert fig.get_axes()[0].get_title() == 'PSP/ISOIS-EPILO 65.9 - 100.5 keV electrons\n5min averaging, viewing: 7'
    # viewing "3", combined channels
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='3', background_range=background_range, channels=[0, 4], resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (198,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2021-10-28 15:33:14.991967')
    assert onset_found
    assert peak_time == pd.Timestamp('2021-10-28 17:48:14.991967')
    assert fig.get_axes()[0].get_title() == 'PSP/ISOIS-EPILO 10.0 - 100.5 keV electrons\n5min averaging, viewing: 3'


def test_onset_Wind_3DP_p_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 29)
    lpath = f"{os.getcwd()}/data/"
    Event1 = Event(spacecraft='Wind', sensor='3DP', data_level='l2', species='protons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    # viewng "sector 3"
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='sector 3', background_range=background_range, channels=4, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2021-10-28 16:27:35.959000')
    assert onset_found
    assert peak_time == pd.Timestamp('2021-10-28 20:17:35.959000')
    assert fig.get_axes()[0].get_title() == 'WIND/3DP 385.96 - 716.78 keV protons\n5min averaging, viewing: SECTOR 3'
    # viewing "omnidirectional"
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='omnidirectional', background_range=background_range, channels=4, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2021-10-28 16:27:42.224000')
    assert onset_found
    assert peak_time == pd.Timestamp('2021-10-28 21:07:42.224000')
    assert fig.get_axes()[0].get_title() == 'WIND/3DP 385.96 - 716.78 keV protons\n5min averaging, viewing: OMNIDIRECTIONAL'
    # no channel combination inlcuded for Wind/3DP, yet


def test_onset_Wind_3DP_e_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 29)
    lpath = f"{os.getcwd()}/data/"
    Event1 = Event(spacecraft='Wind', sensor='3DP', data_level='l2', species='electrons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    #
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='sector 3', background_range=background_range, channels=4, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2021-10-28 16:12:35.959000')
    assert onset_found
    assert peak_time == pd.Timestamp('2021-10-28 21:52:35.959000')
    assert fig.get_axes()[0].get_title() == 'WIND/3DP 127.06 - 235.96 keV electrons\n5min averaging, viewing: SECTOR 3'
    #
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='omnidirectional', background_range=background_range, channels=4, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2021-10-28 16:07:42.224000')
    assert onset_found
    assert peak_time == pd.Timestamp('2021-10-28 21:52:42.224000')
    assert fig.get_axes()[0].get_title() == 'WIND/3DP 127.06 - 235.96 keV electrons\n5min averaging, viewing: OMNIDIRECTIONAL'
    # no channel combination inlcuded for Wind/3DP, yet


def test_onset_STEREOB_HET_p_online():
    startdate = datetime.date(2006, 12, 13)
    enddate = datetime.date(2006, 12, 14)
    lpath = f"{os.getcwd()}/data/"
    Event1 = Event(spacecraft='STEREO-B', sensor='HET', data_level='l2', species='protons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    background_range = (datetime.datetime(2006, 12, 13, 0, 0, 0), datetime.datetime(2006, 12, 13, 2, 0, 0))
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing=None, background_range=background_range, channels=[5, 8], resample_period="5min", yscale='log', cusum_window=30)

    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2006-12-13 03:18:04')
    assert onset_found
    assert peak_time == pd.Timestamp('2006-12-13 09:53:04')
    assert fig.get_axes()[0].get_title() == 'STB/HET 26.3 - 40.5 MeV protons\n5min averaging'


def test_onset_STEREOB_HET_e_online():
    startdate = datetime.date(2006, 12, 13)
    enddate = datetime.date(2006, 12, 14)
    lpath = f"{os.getcwd()}/data/"
    Event1 = Event(spacecraft='STEREO-B', sensor='HET', data_level='l2', species='electrons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    background_range = (datetime.datetime(2006, 12, 13, 0, 0, 0), datetime.datetime(2006, 12, 13, 2, 0, 0))
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing=None, background_range=background_range, channels=[1], resample_period="5min", yscale='log', cusum_window=30)

    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2006-12-13 02:43:04')
    assert onset_found
    assert peak_time == pd.Timestamp('2006-12-13 04:53:04')
    assert fig.get_axes()[0].get_title() == 'STB/HET 1.4 - 2.8 MeV electrons\n5min averaging'


def test_onset_STEREOA_SEPT_p_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 28)
    lpath = f"{os.getcwd()}/data/"
    Event1 = Event(spacecraft='STEREO-A', sensor='SEPT', data_level='l2', species='ions', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='north', background_range=background_range, channels=[5, 8], resample_period="5min", yscale='log', cusum_window=30)

    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2021-10-28 15:53:27.974418944')
    assert onset_found
    assert peak_time == pd.Timestamp('2021-10-28 17:18:27.974418944')
    assert fig.get_axes()[0].get_title() == 'STA/SEPT 110-174.6 keV protons\n5min averaging, viewing: NORTH'


def test_onset_STEREOA_SEPT_e_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 28)
    lpath = f"{os.getcwd()}/data/"
    Event1 = Event(spacecraft='STEREO-A', sensor='SEPT', data_level='l2', species='electrons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='asun', background_range=background_range, channels=[8], resample_period="5min", yscale='log', cusum_window=30)

    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2021-10-28 15:43:27.974418944')
    assert onset_found
    assert peak_time == pd.Timestamp('2021-10-28 18:58:27.974418944')
    assert fig.get_axes()[0].get_title() == 'STA/SEPT 125-145 keV electrons\n5min averaging, viewing: ASUN'


def test_onset_SOHO_EPHIN_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 28)
    lpath = f"{os.getcwd()}/data/"
    Event1 = Event(spacecraft='SOHO', sensor='EPHIN', data_level='l2', species='electrons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing=None, background_range=background_range, channels=150, resample_period="5min", yscale='log', cusum_window=30)

    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2021-10-28 15:53:42.357000')
    assert onset_found
    assert peak_time == pd.Timestamp('2021-10-28 22:18:42.357000')  # pd.Timestamp('2021-10-29 04:53:42.357000')
    assert fig.get_axes()[0].get_title() == 'SOHO/EPHIN 0.25 - 0.7 MeV electrons\n5min averaging'
    # no channel combination inlcuded for SOHO/EPHIN electrons, yet


def test_onset_SOHO_ERNE_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 29)
    lpath = f"{os.getcwd()}/data/"
    Event1 = Event(spacecraft='SOHO', sensor='ERNE-HED', data_level='l2', species='protons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing=None, background_range=background_range, channels=3, resample_period="5min", yscale='log', cusum_window=30)

    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2021-10-28 16:53:05.357000')
    assert onset_found
    assert peak_time == pd.Timestamp('2021-10-28 22:53:05.357000')
    assert fig.get_axes()[0].get_title() == 'SOHO/ERNE 25.0 - 32.0 MeV protons\n5min averaging'


def test_onset_SOHO_ERNE_offline():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 29)
    fullpath = get_pkg_data_filename('data/test/soho_erne-hed_l2-1min_20211028_v01.cdf', package='seppy')
    lpath = Path(fullpath).parent.as_posix()
    Event1 = Event(spacecraft='SOHO', sensor='ERNE-HED', data_level='l2', species='protons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing=None, background_range=background_range, channels=[1, 3], resample_period="5min", yscale='log', cusum_window=30)

    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5] == pd.Timestamp('2021-10-28 16:53:05.357000')
    assert onset_found
    assert peak_time == pd.Timestamp('2021-10-28 22:53:05.357000')
    assert fig.get_axes()[0].get_title() == 'SOHO/ERNE 16.0 - 32.0 MeV protons\n5min averaging'


def test_dynamic_spectrum_SOHO_ERNE_offline():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 29)
    fullpath = get_pkg_data_filename('data/test/soho_erne-hed_l2-1min_20211028_v01.cdf', package='seppy')
    lpath = Path(fullpath).parent.as_posix()
    radio_spacecraft = None  # use ('ahead', 'STEREO-A') if #27 is fixed and radio files can be provided offline
    Event1 = Event(spacecraft='SOHO', sensor='ERNE-HED', data_level='l2', species='protons', start_date=startdate, end_date=enddate, data_path=lpath, radio_spacecraft=radio_spacecraft)
    Event1.dynamic_spectrum(view=None)

    assert Event1.fig.get_axes()[0].get_title() == 'SOHO/ERNE protons, 2021-10-28'
