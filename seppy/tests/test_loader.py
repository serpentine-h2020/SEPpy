import datetime as dt
import numpy as np
import pandas as pd
import pytest
import requests
from astropy.utils.data import get_pkg_data_filename
from pathlib import Path
from seppy.loader.bepi import bepi_sixsp_l3_loader
from seppy.loader.juice import juice_radem_load
from seppy.loader.psp import psp_isois_load
from seppy.loader.soho import soho_load
from seppy.loader.solo import mag_load
from seppy.loader.stereo import stereo_load
from seppy.loader.wind import wind3dp_load, wind3dp_single_download
from unittest.mock import patch


def test_bepi_sixs_load_online():
    startdate = dt.datetime(2020, 10, 9, 12, 0)
    df, meta = bepi_sixsp_l3_loader(startdate=startdate, resample="10min", path=None, pos_timestamp='center')
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (0, 309)
    #
    startdate = dt.datetime(2020, 10, 19, 12, 0)
    df, meta = bepi_sixsp_l3_loader(startdate=startdate, resample="10min", path=None, pos_timestamp='center')
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (78, 309)
    assert df['Side2_P2'].mean() == pytest.approx(np.float64(0.3066333333333333))
    assert len(meta) == 48
    assert meta['Side0_Electron_Bins_str']['E4'] == '278 keV'


def test_juice_radem_no_files_downloaded():
    df, energies, metadata = juice_radem_load(startdate=dt.datetime(2020, 1, 1), enddate=dt.datetime(2020, 1, 1), path="/tmp")

    # Assert: empty dataframe and empty dicts
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert energies == {}
    assert metadata == {}


def test_juice_radem_load_without_resample():
    df, energies, metadata = juice_radem_load(startdate=dt.datetime(2025, 1, 1), enddate=dt.datetime(2025, 1, 1), resample=None, path=None)
    assert "TIME_OBT" not in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["TIME_UTC"])
    assert "PROTONS_4" in df.columns
    assert df.shape == (1440, 64)
    assert df['PROTONS_5'].sum() == 51
    assert energies['LABEL_PROTONS'][0] == 'P-Stack_Bin_1'
    assert metadata['PROTONS']['FILLVAL'] == 4294967295


def test_juice_radem_load_with_resample():
    df, energies, metadata = juice_radem_load(startdate=dt.datetime(2025, 1, 1), enddate=dt.datetime(2025, 1, 1), resample='1h', path=None, pos_timestamp="start")
    assert "TIME_OBT" not in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["TIME_UTC"])
    assert "PROTONS_4" in df.columns
    assert df.shape == (24, 64)
    assert df['PROTONS_5'].sum() == pytest.approx(np.float64(0.8500000000000001))
    assert energies['LABEL_PROTONS'][0] == 'P-Stack_Bin_1'
    assert metadata['PROTONS']['FILLVAL'] == 4294967295
    assert df.keys().to_list() == ['CONFIGURATION_ID', 'CUSTOM_0', 'CUSTOM_1', 'CUSTOM_2', 'CUSTOM_3',
                                   'CUSTOM_4', 'CUSTOM_5', 'CUSTOM_6', 'CUSTOM_7', 'CUSTOM_8', 'CUSTOM_9',
                                   'CUSTOM_10', 'CUSTOM_11', 'DD_0', 'DD_1', 'DD_2', 'DD_3', 'DD_4',
                                   'DD_5', 'DD_6', 'DD_7', 'DD_8', 'DD_9', 'DD_10', 'DD_11', 'DD_12',
                                   'DD_13', 'DD_14', 'DD_15', 'DD_16', 'DD_17', 'DD_18', 'DD_19', 'DD_20',
                                   'DD_21', 'DD_22', 'DD_23', 'DD_24', 'DD_25', 'DD_26', 'DD_27', 'DD_28',
                                   'DD_29', 'DD_30', 'ELECTRONS_0', 'ELECTRONS_1', 'ELECTRONS_2',
                                   'ELECTRONS_3', 'ELECTRONS_4', 'ELECTRONS_5', 'ELECTRONS_6',
                                   'ELECTRONS_7', 'HEAVY_IONS_0', 'HEAVY_IONS_1', 'INTEGRATION_TIME',
                                   'PROTONS_0', 'PROTONS_1', 'PROTONS_2', 'PROTONS_3', 'PROTONS_4',
                                   'PROTONS_5', 'PROTONS_6', 'PROTONS_7', 'TIME_UTC']


def test_psp_load_online():
    df, meta = psp_isois_load(dataset='PSP_ISOIS-EPIHI_L2-HET-RATES60', startdate="2021/05/31",
                              enddate="2021/06/01", path=None, resample=None)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (48, 3244)
    assert meta['H_ENERGY_LABL'].flatten()[0] == '  6.7 -   8.0 MeV'
    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df['B_H_Uncertainty_14'])) == 48
    #
    df2, meta2 = psp_isois_load(dataset='PSP_ISOIS-EPILO_L2-PE', startdate="2021/05/31",
                                enddate="2021/06/01", epilo_channel='F', path=None, resample="1min")
    assert isinstance(df2, pd.DataFrame)
    assert df2.shape == (57, 410)
    assert meta2['Electron_ChanF_Energy']['Electron_ChanF_Energy_E0_P0'] == np.float32(130.09998)
    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df2['Electron_CountRate_ChanF_E47_P7'])) == 57
    #
    df3, meta3 = psp_isois_load(dataset='PSP_ISOIS-EPILO_L2-IC', startdate="2021/05/31",
                                enddate="2021/06/01", epilo_channel='P', path=None, resample="1min")
    assert isinstance(df3, pd.DataFrame)
    assert df3.shape == (57, 11690)
    assert meta3['H_ChanP_Energy']['H_ChanP_Energy_E0_P0'] == np.float32(49.82931)
    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df3['H_Flux_ChanP_E46_P79'])) == 57
    #
    df4, meta4 = psp_isois_load(dataset='PSP_ISOIS-EPIHI_L2-LET1-RATES60', startdate="2021/05/31",
                                enddate="2021/06/01", path=None, resample="1min")
    assert isinstance(df4, pd.DataFrame)
    assert df4.shape == (49, 5728)
    assert meta4['H_ENERGY_LABL'].flatten()[0] == '  0.6 -   0.7 MeV'
    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df4['A_He_Flux_1'])) == 49
    #
    df5, meta5 = psp_isois_load(dataset='PSP_ISOIS-EPIHI_L2-LET2-RATES60', startdate="2021/05/31",
                                enddate="2021/06/01", path=None, resample="1min")
    assert isinstance(df5, pd.DataFrame)
    assert df5.shape == (49, 2770)
    assert meta5['H_ENERGY_LABL'].flatten()[0] == '  0.6 -   0.7 MeV'
    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df5['LET2_C_PA'])) == 4


# deactivate testing of PSP offline loading bc. the version is changing so often (JG 2024/03/26)
# def test_psp_load_offline():
#     # offline data files need to be replaced if data "version" is updated!
#     fullpath = get_pkg_data_filename('data/test/psp_isois-epihi_l2-het-rates60_20210531_v19.cdf', package='seppy')
#     path = Path(fullpath).parent.as_posix()
#     df, meta = psp_isois_load(dataset='PSP_ISOIS-EPIHI_L2-HET-RATES60', startdate="2021/05/31",
#                               enddate="2021/06/01", path=path, resample="1min")
#     assert isinstance(df, pd.DataFrame)
#     assert df.shape == (48, 136)
#     assert meta['H_ENERGY_LABL'][0][0] == '  6.7 -   8.0 MeV'
#     # Check that fillvals are replaced by NaN
#     assert np.sum(np.isnan(df['B_H_Uncertainty_14'])) == 48


def test_soho_ephin_load_online():
    df, meta = soho_load(dataset='SOHO_COSTEP-EPHIN_L2-1MIN', startdate="2021/04/16", enddate="2021/04/16",
                         path=None, resample="2min", pos_timestamp='center')
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (573, 14)
    assert meta['energy_labels']['E1300'] == '0.67 - 10.4 MeV'
    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df['E1300'])) == 219


def test_soho_ephin_load_offline():
    fullpath = get_pkg_data_filename('data/test/epi21106.rl2', package='seppy')
    path = Path(fullpath).parent.as_posix()
    df, meta = soho_load(dataset='SOHO_COSTEP-EPHIN_L2-1MIN', startdate="2021/04/16", enddate="2021/04/16",
                         path=path, resample="2min", pos_timestamp=None)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (573, 14)
    assert meta['energy_labels']['E1300'] == '0.67 - 10.4 MeV'
    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df['E1300'])) == 219


def test_soho_erne_hed_load_online():
    df, meta = soho_load(dataset='SOHO_ERNE-HED_L2-1MIN', startdate="2021/04/16", enddate="2021/04/17",
                         path=None, resample="1min", pos_timestamp='center')
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1145, 41)
    assert meta['channels_dict_df_p']['ch_strings'].iloc[9] == '100 - 130 MeV'
    assert df['PHC_9'].sum() == 1295.0


def test_soho_erne_led_load_online():
    df, meta = soho_load(dataset='SOHO_ERNE-LED_L2-1MIN', startdate="2002/04/16", enddate="2002/04/17",
                         path=None, resample="1min", pos_timestamp='center')
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1438, 41)
    assert meta['channels_dict_df_p']['ch_strings'].iloc[9] == '10  - 13  MeV'
    assert df['PLC_9'].sum() == 155.0


def test_solo_mag_load_online():
    df = mag_load("2021/07/12", "2021/07/13", level='l2', data_type='normal-1-minute', frame='rtn', path=None)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1437, 7)
    assert np.sum(np.isnan(df['B_RTN_0'])) == 64


def test_solo_mag_load_offline():
    # offline data files need to be replaced if data "version" is updated!
    fullpath = get_pkg_data_filename('data/test/solo_l2_mag-rtn-normal-1-minute_20210712_v01.cdf', package='seppy')
    path = Path(fullpath).parent.as_posix()
    df = mag_load("2021/07/12", "2021/07/13", level='l2', data_type='normal-1-minute', frame='rtn', path=path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1437, 7)
    assert np.sum(np.isnan(df['B_RTN_0'])) == 64


def test_stereo_het_load_online():
    df, meta = stereo_load(instrument="HET", startdate="2021/10/28", enddate="2021/10/29",
                           path=None, resample="1min", pos_timestamp='center')
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1440, 28)
    assert meta['Proton_Bins_Text'].flatten()[0] == '13.6 - 15.1 MeV '
    assert np.sum(np.isnan(df['Electron_Flux_0'])) == 0


def test_stereo_het_load_offline():
    # offline data files need to be replaced if data "version" is updated!
    fullpath = get_pkg_data_filename('data/test/sta_l1_het_20211028_v01.cdf', package='seppy')
    path = Path(fullpath).parent.as_posix()
    df, meta = stereo_load(instrument="HET", startdate="2021/10/28", enddate="2021/10/29",
                           path=path, resample="1min", pos_timestamp=None)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1440, 28)
    assert meta['Proton_Bins_Text'].flatten()[0] == '13.6 - 15.1 MeV '
    assert np.sum(np.isnan(df['Electron_Flux_0'])) == 0


def test_stereo_sept_load_online():
    df, meta = stereo_load(instrument="SEPT", startdate="2006/11/14", enddate="2006/11/14",
                           path=None, resample="1min", pos_timestamp='center', sept_viewing='north')
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (371, 30)
    assert meta['channels_dict_df_e'].ch_strings[meta['channels_dict_df_e'].index==2].values[0] == '45.0-55.0 keV'
    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df['ch_2'])) == 371


def test_stereo_sept_load_offline():
    # offline data files need to be replaced if data "version" is updated!
    fullpath = get_pkg_data_filename('data/test/sept_ahead_ele_sun_2006_318_1min_l2_v03.dat', package='seppy')
    path = Path(fullpath).parent.as_posix()
    df, meta = stereo_load(instrument="SEPT", startdate="2006/11/14", enddate="2006/11/14",
                           path=path, resample="1min", pos_timestamp=None, sept_viewing='sun')
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (371, 30)
    assert meta['channels_dict_df_e'].ch_strings[meta['channels_dict_df_e'].index==2].values[0] == '45.0-55.0 keV'
    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df['ch_2'])) == 371


def test_wind3dp_load_online():
    df, meta = wind3dp_load(dataset="WI_SFPD_3DP",
                            startdate="2021/04/15",
                            enddate="2021/04/17",
                            resample='1min',
                            multi_index=True,
                            path=None)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2880, 76)
    assert meta['FLUX_LABELS'].flatten()[0] == 'ElecNoFlux_Ch1_Often~27keV '
    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df['FLUX_E0', 'FLUX_E0_P0'])) == 129


def test_wind3dp_load_offline():
    # offline data files need to be replaced if data "version" is updated!
    fullpath = get_pkg_data_filename('data/test/wi_sfsp_3dp_20200213_v01.cdf', package='seppy')
    path = Path(fullpath).parent.as_posix()
    df, meta = wind3dp_load(dataset="WI_SFSP_3DP",
                            startdate="2020/02/13",
                            enddate="2020/02/14",
                            resample=None,
                            multi_index=False,
                            path=path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (897, 15)
    assert meta['FLUX_LABELS'].flatten()[0] == 'ElecNoFlux_Ch1_Often~27keV '
    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df['FLUX_0'])) == 352


def test_wind3dp_single_download_exceptions():
    """
    Test wind3dp_single_download with mocked request exceptions.
    """
    wind_file = 'wi_sfsp_3dp_20220602_v01.cdf'

    with patch('requests.get', side_effect=requests.exceptions.ReadTimeout):
        downloaded_file = wind3dp_single_download(wind_file)
        assert downloaded_file == []

    with patch('requests.get', side_effect=requests.exceptions.Timeout):
        downloaded_file = wind3dp_single_download(wind_file)
        assert downloaded_file == []

    with patch('requests.get', side_effect=requests.exceptions.HTTPError):
        with pytest.raises(requests.exceptions.HTTPError):
            downloaded_file = wind3dp_single_download(wind_file)
