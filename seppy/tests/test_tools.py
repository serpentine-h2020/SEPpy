from astropy.utils.data import get_pkg_data_filename
from pathlib import Path
from seppy.loader.stereo import stereo_load
from seppy.tools import Event
from seppy.util import jupyterhub_data_path, resample_df
import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import pytest


"""
Install dependencies for tests:
pip install flake8 pytest pytest-doctestplus pytest-cov pytest-mpl

To create/update the baseline images, run the following command from the base package dir:
pytest --mpl-generate-path=seppy/tests/baseline seppy/tests/test_tools.py

To run the tests locally, go to the base directory of the repository and run:
pytest -ra --mpl --mpl-baseline-path=baseline --mpl-baseline-relative --mpl-generate-summary=html seppy/tests/test_tools.py
"""


# switch to non-plotting matplotlib backend to avoid showing all the figures:
plt.switch_backend("Agg")


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_onset_spectrum_tsa_SOLO_STEP_ions_old_data_online():
    startdate = datetime.date(2020, 9, 21)
    enddate = datetime.date(2020, 9, 21)
    lpath = f"{os.getcwd()}{os.sep}data"
    lpath = jupyterhub_data_path(lpath)
    background_range = (datetime.datetime(2020, 9, 21, 0, 0, 0), datetime.datetime(2020, 9, 21, 2, 0, 0))
    #
    # ions
    Event1 = Event(spacecraft='Solar Orbiter', sensor='STEP', viewing='Pixel averaged', data_level='l2', species='ions', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    # Pixel averaged
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='Pixel averaged', background_range=background_range, channels=1, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5].isoformat().split('.')[0] == "2020-09-21T17:07:37"
    assert onset_found
    assert peak_time.isoformat().split('.')[0] == '2020-09-21T17:57:37'
    assert fig.get_axes()[0].get_title() == 'SOLO/STEP 0.0060 - 0.0091 MeV/n protons\n5min averaging, viewing: PIXEL AVERAGED'
    # Pixel 8 - check that calculation is stopped bc. this data is not implemented correctly!
    check = False
    try:
        flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='Pixel 8', background_range=background_range, channels=1, resample_period="5min", yscale='log', cusum_window=30)
    except Warning:
        check = True
    assert check

    # TODO: deactivated, as this function is deactivated atm:
    # test dynamic spectrum:
    # Event1.dynamic_spectrum(view='Pixel averaged')
    # assert Event1.fig.get_axes()[0].get_title() == 'SOLO/STEP (Pixel averaged) ions, 2020-09-21'

    # test tsa plot:
    plt.close('all')  # in order to pick the right figure, make sure all previous are closed
    Event1.tsa_plot('Pixel averaged', selection=(0, 4, 1), resample='1min')
    assert plt.figure(1).get_axes()[0].get_title() == 'Solar Orbiter STEP, ions'

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_onset_spectrum_tsa_SOLO_STEP_ions_new_data_online():
    startdate = datetime.date(2022, 1, 9)
    enddate = datetime.date(2022, 1, 9)
    lpath = f"{os.getcwd()}{os.sep}data"
    lpath = jupyterhub_data_path(lpath)
    background_range = (datetime.datetime(2022, 1, 9, 10, 0, 0), datetime.datetime(2022, 1, 9, 12, 0, 0))
    # ions
    Event1 = Event(spacecraft='Solar Orbiter', sensor='STEP', viewing='Pixel averaged', data_level='l2', species='ions', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    # Pixel averaged
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='Pixel averaged', background_range=background_range, channels=1, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (164,)
    assert len(onset_stats) == 6
    assert isinstance(onset_stats[5], pd._libs.tslibs.nattype.NaTType)  # onset_stats[5].isoformat().split('.')[0] == '2021-10-28T16:07:30'
    assert not onset_found
    assert peak_time.isoformat().split('.')[0] == '2022-01-09T01:32:31'
    assert fig.get_axes()[0].get_title() == 'SOLO/STEP 0.0061 - 0.0091 MeV protons\n5min averaging, viewing: PIXEL AVERAGED'
    # Pixel 8
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='Pixel 8', background_range=background_range, channels=1, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (164,)
    assert len(onset_stats) == 6
    assert isinstance(onset_stats[5], pd._libs.tslibs.nattype.NaTType)  # onset_stats[5].isoformat().split('.')[0] == '2021-10-28T16:12:30'
    assert not onset_found
    assert peak_time.isoformat().split('.')[0] == '2022-01-09T00:02:31'
    assert fig.get_axes()[0].get_title() == 'SOLO/STEP 0.0061 - 0.0091 MeV protons\n5min averaging, viewing: PIXEL 8'

    # TODO: deactivated, as this function is deactivated atm:
    # test dynamic spectrum:
    # Event1.dynamic_spectrum(view='Pixel averaged')
    # assert Event1.fig.get_axes()[0].get_title() == 'SOLO/STEP (Pixel averaged) ions, 2022-01-09'
    # Event1.dynamic_spectrum(view='Pixel 8')
    # assert Event1.fig.get_axes()[0].get_title() == 'SOLO/STEP (Pixel 8) ions, 2022-01-09'

    # test tsa plot:
    plt.close('all')  # in order to pick the right figure, make sure all previous are closed
    Event1.tsa_plot('Pixel 8', selection=(0, 4, 1), resample='1min')
    assert plt.figure(1).get_axes()[0].get_title() == 'Solar Orbiter STEP, ions'

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_onset_spectrum_tsa_SOLO_HET_online():
    startdate = datetime.date(2022, 11, 8)
    enddate = datetime.date(2022, 11, 8)
    lpath = f"{os.getcwd()}{os.sep}data"
    lpath = jupyterhub_data_path(lpath)
    background_range = (datetime.datetime(2022, 11, 8, 0, 0, 0), datetime.datetime(2022, 11, 8, 1, 0, 0))
    # viewing "sun", single channel, protons
    Event1 = Event(spacecraft='Solar Orbiter', sensor='HET', viewing='sun', data_level='l2', species='protons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='sun', background_range=background_range, channels=1, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (73,)
    assert len(onset_stats) == 6
    assert isinstance(onset_stats[5], pd._libs.tslibs.nattype.NaTType)  # onset_stats[5] == pd.Timestamp('2021-10-28 15:31:59.492059')
    assert not onset_found
    assert peak_time.isoformat().split('.')[0] == '2022-11-08T17:58:09'
    assert fig.get_axes()[0].get_title() == 'SOLO/HET 7.3540 - 7.8900 MeV protons\n5min averaging, viewing: SUN'
    # viewing "north", combined channel, electrons
    Event1 = Event(spacecraft='Solar Orbiter', sensor='HET', viewing='sun', data_level='l2', species='electrons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='north', background_range=background_range, channels=[0, 3], resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (73,)
    assert len(onset_stats) == 6
    assert isinstance(onset_stats[5], pd._libs.tslibs.nattype.NaTType)  # onset_stats[5] == pd.Timestamp('2021-10-28 15:31:59.492059')
    assert not onset_found
    assert peak_time.isoformat().split('.')[0] == '2022-11-08T22:27:56'
    assert fig.get_axes()[0].get_title() == 'SOLO/HET 0.4533 - 18.8300 MeV electrons\n5min averaging, viewing: NORTH'

    # test dynamic spectrum:
    Event1.dynamic_spectrum(view='sun')
    assert Event1.fig.get_axes()[0].get_title() == 'SOLO/HET (sun) electrons, 2022-11-08'

    # test tsa plot:
    plt.close('all')  # in order to pick the right figure, make sure all previous are closed
    Event1.tsa_plot('north', selection=None, resample='1min')
    assert plt.figure(1).get_axes()[0].get_title() == 'Solar Orbiter HET, electrons'

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_onset_spectrum_tsa_SOLO_EPT_online():
    startdate = datetime.date(2022, 6, 6)
    enddate = datetime.date(2022, 6, 6)
    lpath = f"{os.getcwd()}{os.sep}data"
    lpath = jupyterhub_data_path(lpath)
    background_range = (datetime.datetime(2022, 6, 6, 0, 0, 0), datetime.datetime(2022, 6, 6, 1, 0, 0))
    # viewing "sun", single channel, ions
    Event1 = Event(spacecraft='Solar Orbiter', sensor='EPT', viewing='sun', data_level='l2', species='ions', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='sun', background_range=background_range, channels=4, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert isinstance(onset_stats[5], pd._libs.tslibs.nattype.NaTType)  # onset_stats[5] == pd.Timestamp('2021-10-28 15:31:59.492059')
    assert not onset_found
    assert peak_time.isoformat().split('.')[0] == '2022-06-06T01:02:31'
    assert fig.get_axes()[0].get_title() == 'SOLO/EPT 0.0608 - 0.0678 MeV protons\n5min averaging, viewing: SUN'
    # viewing "north", combined channel, electrons
    Event1 = Event(spacecraft='Solar Orbiter', sensor='EPT', viewing='sun', data_level='l2', species='electrons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='north', background_range=background_range, channels=[5, 10], resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert isinstance(onset_stats[5], pd._libs.tslibs.nattype.NaTType)  # onset_stats[5] == pd.Timestamp('2021-10-28 15:31:59.492059')
    assert not onset_found
    assert peak_time.isoformat().split('.')[0] == '2022-06-06T01:32:31'
    assert fig.get_axes()[0].get_title() == 'SOLO/EPT 0.0439 - 0.0682 MeV electrons\n5min averaging, viewing: NORTH'

    # test dynamic spectrum:
    Event1.dynamic_spectrum(view='sun')
    assert Event1.fig.get_axes()[0].get_title() == 'SOLO/EPT (sun) electrons, 2022-06-06'

    # test tsa plot:
    plt.close('all')  # in order to pick the right figure, make sure all previous are closed
    Event1.tsa_plot('sun', selection=(0, 4, 1), resample='1min')
    assert plt.figure(1).get_axes()[0].get_title() == 'Solar Orbiter EPT, electrons'

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_onset_spectrum_tsa_PSP_ISOIS_EPIHI_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 29)
    lpath = f"{os.getcwd()}{os.sep}data"
    lpath = jupyterhub_data_path(lpath)
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    # viewing "A", single channel, electrons
    Event1 = Event(spacecraft='PSP', sensor='isois-epihi', viewing='A', data_level='l2', species='electrons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='A', background_range=background_range, channels=4, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (194,)
    assert len(onset_stats) == 6
    assert onset_stats[5].isoformat().split('.')[0] == '2021-10-28T15:31:59'
    assert onset_found
    assert peak_time.isoformat().split('.')[0] == '2021-10-28T16:06:59'
    assert fig.get_axes()[0].get_title() == 'PSP/ISOIS-EPIHI 0.8 - 1.0 MeV electrons\n5min averaging, viewing: A'
    # viewing "B", combined channel, protons
    Event1 = Event(spacecraft='PSP', sensor='isois-epihi', viewing='B', data_level='l2', species='protons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='B', background_range=background_range, channels=[1, 5], resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (194,)
    assert len(onset_stats) == 6
    assert onset_stats[5].isoformat().split('.')[0] == '2021-10-28T16:46:59'
    assert onset_found
    assert peak_time.isoformat().split('.')[0] == '2021-10-28T19:56:59'
    assert fig.get_axes()[0].get_title() == 'PSP/ISOIS-EPIHI 8.0 - 19.0 MeV protons\n5min averaging, viewing: B'

    # test dynamic spectrum:
    Event1.dynamic_spectrum(view='A')
    assert Event1.fig.get_axes()[0].get_title() == 'PSP/ISOIS-EPIHI (A) protons, 2021-10-28'

    # test tsa plot:
    plt.close('all')  # in order to pick the right figure, make sure all previous are closed
    Event1.tsa_plot('A', selection=(0, 4, 1), resample='1min')
    assert plt.figure(1).get_axes()[0].get_title() == 'Parker Solar Probe ISOIS-EPIHI, protons'

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_onset_spectrum_tsa_PSP_ISOIS_EPILO_e_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 29)
    lpath = f"{os.getcwd()}{os.sep}data"
    lpath = jupyterhub_data_path(lpath)
    Event1 = Event(spacecraft='PSP', sensor='isois-epilo', viewing='7', data_level='l2', species='electrons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    # viewing "7", single channel
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='7', background_range=background_range, channels=4, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (198,)
    assert len(onset_stats) == 6
    assert onset_stats[5].isoformat().split('.')[0] == '2021-10-28T15:33:14'
    assert onset_found
    assert peak_time.isoformat().split('.')[0] == '2021-10-28T16:43:14'
    assert fig.get_axes()[0].get_title() == 'PSP/ISOIS-EPILO 65.9 - 100.5 keV electrons\n5min averaging, viewing: 7'
    # viewing "3", combined channels
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='3', background_range=background_range, channels=[0, 4], resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (198,)
    assert len(onset_stats) == 6
    assert onset_stats[5].isoformat().split('.')[0] == '2021-10-28T15:33:14'
    assert onset_found
    assert peak_time.isoformat().split('.')[0] == '2021-10-28T17:48:14'
    assert fig.get_axes()[0].get_title() == 'PSP/ISOIS-EPILO 10.0 - 100.5 keV electrons\n5min averaging, viewing: 3'

    # test dynamic spectrum:
    Event1.dynamic_spectrum(view='7')
    assert Event1.fig.get_axes()[0].get_title() == 'PSP/ISOIS-EPILO (7) electrons, 2021-10-28'

    # test tsa plot:
    plt.close('all')  # in order to pick the right figure, make sure all previous are closed
    Event1.tsa_plot('3', selection=(0, 4, 1), resample='1min')
    assert plt.figure(1).get_axes()[0].get_title() == 'Parker Solar Probe ISOIS-EPILO, electrons'

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_onset_spectrum_tsa_Wind_3DP_p_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 29)
    lpath = f"{os.getcwd()}{os.sep}data"
    lpath = jupyterhub_data_path(lpath)
    Event1 = Event(spacecraft='Wind', sensor='3DP', data_level='l2', viewing="Sector 3", species='protons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    # viewng "sector 3"
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='sector 3', background_range=background_range, channels=4, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5].isoformat().split('.')[0] == "2021-10-28T16:12:35"
    assert onset_found
    assert peak_time.isoformat().split('.')[0] == '2021-10-28T20:17:35'
    assert fig.get_axes()[0].get_title() == 'WIND/3DP 385.96 - 716.78 keV protons\n5min averaging, viewing: SECTOR 3'
    # viewing "omnidirectional"
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='omnidirectional', background_range=background_range, channels=4, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5].isoformat().split('.')[0] == "2021-10-28T16:07:42"
    assert onset_found
    assert peak_time.isoformat().split('.')[0] == '2021-10-28T21:07:42'
    assert fig.get_axes()[0].get_title() == 'WIND/3DP 385.96 - 716.78 keV protons\n5min averaging, viewing: OMNIDIRECTIONAL'
    # no channel combination inlcuded for Wind/3DP, yet

    # test dynamic spectrum:
    Event1.dynamic_spectrum(view='sector 3')
    assert Event1.fig.get_axes()[0].get_title() == 'WIND/3DP (sector 3) protons, 2021-10-28'

    # test tsa plot:
    plt.close('all')  # in order to pick the right figure, make sure all previous are closed
    Event1.tsa_plot('omnidirectional', selection=(0, 4, 1), resample=None)
    assert plt.figure(1).get_axes()[0].get_title() == 'Wind 3DP, protons'

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_onset_spectrum_tsa_Wind_3DP_e_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 29)
    lpath = f"{os.getcwd()}{os.sep}data"
    lpath = jupyterhub_data_path(lpath)
    Event1 = Event(spacecraft='Wind', sensor='3DP', viewing="Sector 3", data_level='l2', species='electrons', start_date=startdate, end_date=enddate, data_path=lpath)  # TODO: radio_spacecraft=('wind', 'WIND')
    print(Event1.print_energies())
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    #
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='sector 3', background_range=background_range, channels=4, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5].isoformat().split('.')[0] == "2021-10-28T15:57:35"
    assert onset_found
    assert peak_time.isoformat().split('.')[0] == '2021-10-28T21:52:35'
    assert fig.get_axes()[0].get_title() == 'WIND/3DP 127.06 - 235.96 keV electrons\n5min averaging, viewing: SECTOR 3'
    #
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='omnidirectional', background_range=background_range, channels=4, resample_period="5min", yscale='log', cusum_window=30)
    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5].isoformat().split('.')[0] == "2021-10-28T15:57:42"
    assert onset_found
    assert peak_time.isoformat().split('.')[0] == '2021-10-28T21:52:42'
    assert fig.get_axes()[0].get_title() == 'WIND/3DP 127.06 - 235.96 keV electrons\n5min averaging, viewing: OMNIDIRECTIONAL'
    # no channel combination inlcuded for Wind/3DP, yet

    # test dynamic spectrum:
    Event1.dynamic_spectrum(view='omnidirectional')
    assert Event1.fig.get_axes()[0].get_title() == 'WIND/3DP (omnidirectional) electrons, 2021-10-28'

    # test tsa plot:
    plt.close('all')  # in order to pick the right figure, make sure all previous are closed
    Event1.tsa_plot('sector 3', selection=(0, 4, 1), resample=None)
    assert plt.figure(1).get_axes()[0].get_title() == 'Wind 3DP, electrons'

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_onset_spectrum_tsa_STEREOB_HET_p_online():
    startdate = datetime.date(2006, 12, 13)
    enddate = datetime.date(2006, 12, 14)
    lpath = f"{os.getcwd()}{os.sep}data"
    lpath = jupyterhub_data_path(lpath)
    Event1 = Event(spacecraft='STEREO-B', sensor='HET', data_level='l2', species='protons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    background_range = (datetime.datetime(2006, 12, 13, 0, 0, 0), datetime.datetime(2006, 12, 13, 2, 0, 0))
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing=None, background_range=background_range, channels=[5, 8], resample_period="5min", yscale='log', cusum_window=30)

    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5].isoformat().split('.')[0] == "2006-12-13T03:03:04"
    assert onset_found
    assert peak_time.isoformat().split('.')[0] == '2006-12-13T09:53:04'
    assert fig.get_axes()[0].get_title() == 'STB/HET 26.3 - 40.5 MeV protons\n5min averaging'

    # test dynamic spectrum:
    Event1.dynamic_spectrum(view=None)
    assert Event1.fig.get_axes()[0].get_title() == 'STB/HET protons, 2006-12-13'

    # test tsa plot:
    plt.close('all')  # in order to pick the right figure, make sure all previous are closed
    Event1.tsa_plot(None, selection=None, resample=None)
    assert plt.figure(1).get_axes()[0].get_title() == 'STEREO-B HET, protons'

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_onset_spectrum_tsa_STEREOB_HET_e_online():
    startdate = datetime.date(2006, 12, 13)
    enddate = datetime.date(2006, 12, 14)
    lpath = f"{os.getcwd()}{os.sep}data"
    lpath = jupyterhub_data_path(lpath)
    Event1 = Event(spacecraft='STEREO-B', sensor='HET', data_level='l2', species='electrons', start_date=startdate, end_date=enddate, data_path=lpath, radio_spacecraft=('behind', 'STEREO-B'))
    print(Event1.print_energies())
    background_range = (datetime.datetime(2006, 12, 13, 0, 0, 0), datetime.datetime(2006, 12, 13, 2, 0, 0))
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing=None, background_range=background_range, channels=[1], resample_period="5min", yscale='log', cusum_window=30)

    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5].isoformat().split('.')[0] == "2006-12-13T02:38:04"
    assert onset_found
    assert peak_time.isoformat().split('.')[0] == '2006-12-13T04:53:04'
    assert fig.get_axes()[0].get_title() == 'STB/HET 1.4 - 2.8 MeV electrons\n5min averaging'

    # test dynamic spectrum:
    Event1.dynamic_spectrum(view=None)
    assert Event1.fig.get_axes()[0].get_title() == 'Radio & Dynamic Spectrum, STB/HET electrons, 2006-12-13'

    # test tsa plot:
    plt.close('all')  # in order to pick the right figure, make sure all previous are closed
    Event1.tsa_plot(None, selection=None, resample=None)
    assert plt.figure(1).get_axes()[0].get_title() == 'STEREO-B HET, electrons'

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_onset_spectrum_tsa_STEREOA_SEPT_p_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 28)
    lpath = f"{os.getcwd()}{os.sep}data"
    lpath = jupyterhub_data_path(lpath)
    Event1 = Event(spacecraft='STEREO-A', sensor='SEPT', viewing="north", data_level='l2', species='ions', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='north', background_range=background_range, channels=[5, 8], resample_period="5min", yscale='log', cusum_window=30)

    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5].isoformat().split('.')[0] == "2021-10-28T15:48:27"
    assert onset_found
    assert peak_time.isoformat().split('.')[0] == '2021-10-28T17:18:27'
    assert fig.get_axes()[0].get_title() == 'STA/SEPT 110-174.6 keV protons\n5min averaging, viewing: NORTH'

    # test dynamic spectrum:
    Event1.dynamic_spectrum(view="north")
    assert Event1.fig.get_axes()[0].get_title() == 'STA/SEPT (north) protons, 2021-10-28'

    # test tsa plot:
    plt.close('all')  # in order to pick the right figure, make sure all previous are closed
    Event1.tsa_plot(view="north", selection=None, resample=None)
    assert plt.figure(1).get_axes()[0].get_title() == 'STEREO-A SEPT, protons'

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_onset_spectrum_tsa_STEREOA_SEPT_e_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 28)
    lpath = f"{os.getcwd()}{os.sep}data"
    lpath = jupyterhub_data_path(lpath)
    Event1 = Event(spacecraft='STEREO-A', sensor='SEPT', viewing="asun", data_level='l2', species='electrons', start_date=startdate, end_date=enddate, data_path=lpath, radio_spacecraft=('ahead', 'STEREO-A'))
    print(Event1.print_energies())
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='asun', background_range=background_range, channels=[8], resample_period="5min", yscale='log', cusum_window=30)

    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5].isoformat().split('.')[0] == '2021-10-28T15:43:27'
    assert onset_found
    assert peak_time.isoformat().split('.')[0] == '2021-10-28T18:58:27'
    assert fig.get_axes()[0].get_title() == 'STA/SEPT 125-145 keV electrons\n5min averaging, viewing: ASUN'

    # test dynamic spectrum:
    Event1.dynamic_spectrum(view="asun")
    assert Event1.fig.get_axes()[0].get_title() == 'Radio & Dynamic Spectrum, STA/SEPT (asun) electrons, 2021-10-28'

    # test tsa plot:
    plt.close('all')  # in order to pick the right figure, make sure all previous are closed
    Event1.tsa_plot(view="asun", selection=None, resample=None)
    assert plt.figure(1).get_axes()[0].get_title() == 'STEREO-A SEPT, electrons'

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_onset_spectrum_tsa_SOHO_EPHIN_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 28)
    lpath = f"{os.getcwd()}{os.sep}data"
    lpath = jupyterhub_data_path(lpath)
    Event1 = Event(spacecraft='SOHO', sensor='EPHIN', data_level='l2', species='electrons', start_date=startdate, end_date=enddate, data_path=lpath)
    print(Event1.print_energies())
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing=None, background_range=background_range, channels=150, resample_period="5min", yscale='log', cusum_window=30)

    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5].isoformat().split('.')[0] == "2021-10-28T15:28:42"
    assert onset_found
    assert peak_time.isoformat().split('.')[0] == '2021-10-28T22:18:42'  # '2021-10-29T04:53:42.357000')
    assert fig.get_axes()[0].get_title() == 'SOHO/EPHIN 0.25 - 0.7 MeV electrons\n5min averaging'
    # no channel combination inlcuded for SOHO/EPHIN electrons, yet

    # test dynamic spectrum:
    # check = False
    # try:
    #     Event1.dynamic_spectrum(view=None)
    # except Warning:
    #     check = True
    # assert check
    Event1.dynamic_spectrum(view=None)
    assert Event1.fig.get_axes()[0].get_title() == 'SOHO/EPHIN electrons, 2021-10-28'

    # test tsa plot:
    plt.close('all')  # in order to pick the right figure, make sure all previous are closed
    Event1.tsa_plot(None, selection=(0, 4, 1), resample='5min')
    assert plt.figure(1).get_axes()[0].get_title() == 'SOHO EPHIN, electrons'

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_onset_spectrum_tsa_SOHO_ERNE_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 29)
    lpath = f"{os.getcwd()}{os.sep}data"
    lpath = jupyterhub_data_path(lpath)
    Event1 = Event(spacecraft='SOHO', sensor='ERNE-HED', data_level='l2', species='protons', start_date=startdate, end_date=enddate, data_path=lpath, radio_spacecraft=('ahead', 'STEREO-A'))
    print(Event1.print_energies())
    background_range = (datetime.datetime(2021, 10, 28, 10, 0, 0), datetime.datetime(2021, 10, 28, 12, 0, 0))
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing=None, background_range=background_range, channels=3, resample_period="5min", yscale='log', cusum_window=30)

    assert isinstance(flux, pd.Series)
    assert flux.shape == (288,)
    assert len(onset_stats) == 6
    assert onset_stats[5].isoformat().split('.')[0] == "2021-10-28T16:23:05"
    assert onset_found
    assert peak_time.isoformat().split('.')[0] == '2021-10-28T22:53:05'
    assert fig.get_axes()[0].get_title() == 'SOHO/ERNE 25.0 - 32.0 MeV protons\n5min averaging'

    # test dynamic spectrum:
    Event1.dynamic_spectrum(view=None)

    assert Event1.fig.get_axes()[0].get_title() == 'Radio & Dynamic Spectrum, SOHO/ERNE protons, 2021-10-28'

    # test tsa plot:
    plt.close('all')  # in order to pick the right figure, make sure all previous are closed
    Event1.tsa_plot(None, selection=(0, 4, 1), resample='5min')
    assert plt.figure(1).get_axes()[0].get_title() == 'SOHO ERNE, protons'

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_onset_tsa_SOHO_ERNE_offline():
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
    assert onset_stats[5].isoformat().split('.')[0] == "2021-10-28T16:23:05"
    assert onset_found
    assert peak_time.isoformat().split('.')[0] == '2021-10-28T22:53:05'
    assert fig.get_axes()[0].get_title() == 'SOHO/ERNE 16.0 - 32.0 MeV protons\n5min averaging'

    # test tsa plot:
    plt.close('all')  # in order to pick the right figure, make sure all previous are closed
    Event1.tsa_plot(None, selection=(0, 4, 1), resample='5min')
    assert plt.figure(1).get_axes()[0].get_title() == 'SOHO ERNE, protons'

    return fig


def test_dynamic_spectrum_SOHO_ERNE_offline():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 29)
    fullpath = get_pkg_data_filename('data/test/soho_erne-hed_l2-1min_20211028_v01.cdf', package='seppy')
    lpath = Path(fullpath).parent.as_posix()
    radio_spacecraft = None  # use ('ahead', 'STEREO-A') if #27 is fixed and radio files can be provided offline
    Event1 = Event(spacecraft='SOHO', sensor='ERNE-HED', data_level='l2', species='protons', start_date=startdate, end_date=enddate, data_path=lpath, radio_spacecraft=radio_spacecraft)
    Event1.dynamic_spectrum(view=None)

    assert Event1.fig.get_axes()[0].get_title() == 'SOHO/ERNE protons, 2021-10-28'


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_onset_Bepi_SIXS_L2_offline():
    startdate = datetime.date(2023, 7, 19)
    enddate = datetime.date(2023, 7, 19)
    fullpath = get_pkg_data_filename('data/test/20230719_side1.csv', package='seppy')
    lpath = Path(fullpath).parent.as_posix()
    # lpath = '/home/jagies/data/bepi/bc_mpo_sixs/data_csv/cruise/sixs-p/raw'
    Event1 = Event(spacecraft='Bepi', sensor='SIXS-P', data_level='l2', species='electrons', start_date=startdate, end_date=enddate, data_path=lpath, viewing='1')
    background_range = (datetime.datetime(2023, 7, 19, 0, 30, 0), datetime.datetime(2023, 7, 19, 1, 30, 0))
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing='1', background_range=background_range, channels=2, resample_period="1min", yscale='log', cusum_window=30)

    assert isinstance(flux, pd.Series)
    assert flux.shape == (161,)
    assert len(onset_stats) == 6
    #assert isinstance(onset_stats[5], pd._libs.tslibs.nattype.NaTType)
    assert onset_stats[5].isoformat().split('.')[0] == "2023-07-19T02:05:30"
    assert onset_found
    assert peak_time.isoformat().split('.')[0] == '2023-07-19T02:25:30'
    assert fig.get_axes()[0].get_title() == 'BEPI/SIXS-P 106 keV electrons\n1min averaging, viewing: 1'

    return fig


@pytest.mark.parametrize("viewing, species, channels, resample", [('0', 'electrons', [5, 6], '5min'), ('2', 'protons', [8, 9], '10min'), ('1', 'electrons', 2, None)])
@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_onset_Bepi_SIXS_L3_online(viewing, species, channels, resample):
    startdate = datetime.date(2023, 2, 17)
    enddate = datetime.date(2023, 2, 19)
    lpath = f"{os.getcwd()}{os.sep}data"
    lpath = jupyterhub_data_path(lpath)
    Event1 = Event(spacecraft='BepiColombo', sensor='SIXS-P', data_level='l3', species=species, start_date=startdate, end_date=enddate, data_path=lpath, viewing=viewing)
    background_range = (datetime.datetime(2023, 2, 17, 3, 0, 0), datetime.datetime(2023, 2, 17, 19, 0, 0))
    flux, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean = Event1.find_onset(viewing=viewing, background_range=background_range, channels=channels, resample_period=resample, yscale='log', cusum_window=30)

    # assert isinstance(flux, pd.Series)
    # assert flux.shape == (161,)
    # assert len(onset_stats) == 6
    # #assert isinstance(onset_stats[5], pd._libs.tslibs.nattype.NaTType)
    # assert onset_stats[5].isoformat().split('.')[0] == "2023-07-19T02:05:30"
    # assert onset_found
    # assert peak_time.isoformat().split('.')[0] == '2023-07-19T02:25:30'
    # assert fig.get_axes()[0].get_title() == 'BEPI/SIXS 106 keV electrons\n1min averaging, viewing: 1'

    return fig


def test_resample_df_sept_online():
    startdate = datetime.date(2021, 10, 28)
    enddate = datetime.date(2021, 10, 28)
    lpath = f"{os.getcwd()}{os.sep}data"
    lpath = jupyterhub_data_path(lpath)
    df, meta = stereo_load(instrument='SEPT', startdate=startdate, enddate=enddate, spacecraft='STA',
                                             sept_species='p', sept_viewing='sun', path=lpath)
    unc_id = 'err_ch_'
    cols_unc = df.filter(like=unc_id).columns
    #
    resample = '1s'
    with pytest.raises(ValueError, match=f"Your resample option of '{resample}' is smaller than the original data cadence of '0 days 00:01:00'. This is not supported!"):
        df_resampled_auto = resample_df(df, resample, cols_unc='auto')
    #
    resample = '5min'
    df_resampled_none = resample_df(df, resample)
    df_resampled_manu = resample_df(df, resample, cols_unc=cols_unc)
    df_resampled_auto = resample_df(df, resample, cols_unc='auto')
    pd.testing.assert_frame_equal(df_resampled_manu, df_resampled_auto)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df_resampled_none, df_resampled_manu)
    df_resampled_none.index.freqstr == resample
    df_resampled_auto.index.freqstr == resample
    df_resampled_none.err_ch_31.mean() == pytest.approx(0.19295445763888888)
    df_resampled_auto.err_ch_31.mean() == pytest.approx(0.0879893429380724)
    df_resampled_none.err_ch_31.max() == pytest.approx(1.038175)
    df_resampled_auto.err_ch_31.max() == pytest.approx(0.5191970236576863)
    #
    resample = None
    with pytest.raises(ValueError, match="Resample period is set to 'None'. No resampling will be applied."):
        df_resampled_none = resample_df(df, resample)
    #
    return
