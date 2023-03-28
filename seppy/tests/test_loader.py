import numpy as np
import pandas as pd
from astropy.utils.data import get_pkg_data_filename
from pathlib import Path
from seppy.loader.wind import wind3dp_load


def test_wind3dp_load_online():
    df, meta = wind3dp_load(dataset="WI_SFPD_3DP",
                            startdate="2021/04/16",
                            enddate="2021/04/18",
                            resample='1min',
                            multi_index=True,
                            path=None)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2880, 76)
    meta['FLUX_LABELS'][0][0] == 'ElecNoFlux_Ch1_Often~27keV '

    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df['FLUX_E0', 'FLUX_E0_P0'])) == 169


def test_wind3dp_load_offline():
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
    assert meta['FLUX_LABELS'][0][0] == 'ElecNoFlux_Ch1_Often~27keV '

    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df['FLUX_0'])) == 352
