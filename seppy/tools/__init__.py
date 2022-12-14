import copy
import os
import datetime
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as cl
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as const
import sunpy.sun.constants as sconst
from sunpy.coordinates import get_horizons_coord
from matplotlib import rcParams
from matplotlib.dates import DateFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib.offsetbox import AnchoredText
from numpy import sqrt, log, pi
from pandas.tseries.frequencies import to_offset
from seppy.loader.psp import calc_av_en_flux_PSP_EPIHI, calc_av_en_flux_PSP_EPILO, psp_isois_load
from seppy.loader.soho import calc_av_en_flux_ERNE, soho_load
from seppy.loader.solo import epd_load
from seppy.loader.stereo import calc_av_en_flux_HET as calc_av_en_flux_ST_HET
from seppy.loader.stereo import calc_av_en_flux_SEPT, stereo_load
from seppy.loader.wind import wind3dp_load

from IPython.core.display import display

# This is to get rid of this specific warning:
# /home/user/xyz/serpentine/notebooks/sep_analysis_tools/read_swaves.py:96: UserWarning: The input coordinates to pcolormesh are interpreted as
# cell centers, but are not monotonically increasing or decreasing. This may lead to incorrectly calculated cell edges, in which
# case, please supply explicit cell edges to pcolormesh.
# colormesh = ax.pcolormesh( time_arr, freq[::-1], data_arr[::-1], vmin = 0, vmax = 0.5*np.max(data_arr), cmap = 'inferno' )
warnings.filterwarnings("ignore", category=UserWarning)


class Event:

    def __init__(self, start_date, end_date, spacecraft, sensor,
                 species, data_level, data_path, viewing=None, radio_spacecraft=None,
                 threshold=None):

        if spacecraft == "Solar Orbiter":
            spacecraft = "solo"
        if spacecraft == "STEREO-A":
            spacecraft = "sta"
        if spacecraft == "STEREO-B":
            spacecraft = "stb"

        if sensor in ["ERNE-HED"]:
            sensor = "ERNE"

        if species in ("protons", "ions"):
            species = 'p'
        if species == "electrons":
            species = 'e'

        self.start_date = start_date
        self.end_date = end_date
        self.spacecraft = spacecraft.lower()
        self.sensor = sensor.lower()
        self.species = species.lower()
        self.data_level = data_level.lower()
        self.data_path = data_path + os.sep
        self.threshold = threshold
        self.radio_spacecraft = radio_spacecraft  # this is a 2-tuple, e.g., ("ahead", "STEREO-A")
        self.viewing = viewing

        self.radio_files = None

        # placeholding class attributes
        self.flux_series = None
        self.onset_stats = None
        self.onset_found = None
        self.onset = None
        self.peak_flux = None
        self.peak_time = None
        self.fig = None
        self.bg_mean = None
        self.output = {"flux_series": self.flux_series,
                       "onset_stats": self.onset_stats,
                       "onset_found": self.onset_found,
                       "onset": self.onset,
                       "peak_flux": self.peak_flux,
                       "peak_time": self.peak_time,
                       "fig": self.fig,
                       "bg_mean": self.bg_mean
                       }

        # I think it could be worth considering to run self.choose_data(viewing) when the object is created,
        # because now it has to be run inside self.print_energies() to make sure that either
        # self.current_df, self.current_df_i or self.current_df_e exists, because print_energies() needs column
        # names from the dataframe.
        self.load_all_viewing()

        # Download radio cdf files ONLY if asked to
        if self.radio_spacecraft is not None:
            from seppy.tools.swaves import get_swaves
            self.radio_files = get_swaves(start_date, end_date)

    def update_onset_attributes(self, flux_series, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean):
        """
        Method to update onset-related attributes, that are None by default and only have values after analyse() has been run.
        """
        self.flux_series = flux_series
        self.onset_stats = onset_stats
        self.onset_found = onset_found
        self.onset = onset_stats[-1]
        self.peak_flux = peak_flux
        self.peak_time = peak_time
        self.fig = fig
        self.bg_mean = bg_mean

        # also remember to update the dictionary, it won't update automatically
        self.output = {"flux_series": self.flux_series,
                       "onset_stats": self.onset_stats,
                       "onset_found": self.onset_found,
                       "onset": self.onset,
                       "peak_flux": self.peak_flux,
                       "peak_time": self.peak_time,
                       "fig": self.fig,
                       "bg_mean": self.bg_mean
                       }

    def update_viewing(self, viewing):
        if self.spacecraft != "wind":
            self.viewing = viewing
        else:
            # Wind/3DP viewing directions are omnidirectional, section 0, section 1... section n.
            # This catches the number or the word if omnidirectional
            try:
                self.viewing = viewing.split(" ")[-1]

            # AttributeError is cause by initializing Event with spacecraft='Wind' and viewing=None
            except AttributeError:
                self.viewing = '0'  # A placeholder viewing that should not cause any trouble

    # I suggest we at some point erase the arguments ´spacecraft´ and ´threshold´ due to them not being used.
    # `viewing` and `autodownload` are actually the only necessary input variables for this function, the rest
    # are class attributes, and should probably be cleaned up at some point
    def load_data(self, spacecraft, sensor, viewing, data_level,
                  autodownload=True, threshold=None):

        if self.spacecraft == 'solo':
            df_i, df_e, energs = epd_load(sensor=sensor,
                                          viewing=viewing,
                                          level=data_level,
                                          startdate=self.start_date,
                                          enddate=self.end_date,
                                          path=self.data_path,
                                          autodownload=autodownload)

            self.update_viewing(viewing)
            return df_i, df_e, energs

        if self.spacecraft[:2].lower() == 'st':
            if self.sensor == 'sept':
                if self.species in ["p", "i"]:
                    df_i, channels_dict_df_i = stereo_load(instrument=self.sensor,
                                                           startdate=self.start_date,
                                                           enddate=self.end_date,
                                                           spacecraft=self.spacecraft,
                                                           # sept_species=self.species,
                                                           sept_species='p',
                                                           sept_viewing=viewing,
                                                           resample=None,
                                                           path=self.data_path)
                    df_e, channels_dict_df_e = [], []

                    self.update_viewing(viewing)
                    return df_i, df_e, channels_dict_df_i, channels_dict_df_e

                if self.species == "e":
                    df_e, channels_dict_df_e = stereo_load(instrument=self.sensor,
                                                           startdate=self.start_date,
                                                           enddate=self.end_date,
                                                           spacecraft=self.spacecraft,
                                                           # sept_species=self.species,
                                                           sept_species='e',
                                                           sept_viewing=viewing,
                                                           resample=None,
                                                           path=self.data_path)

                    df_i, channels_dict_df_i = [], []

                    self.update_viewing(viewing)
                    return df_i, df_e, channels_dict_df_i, channels_dict_df_e

            if self.sensor == 'het':
                df, meta = stereo_load(instrument=self.sensor,
                                       startdate=self.start_date,
                                       enddate=self.end_date,
                                       spacecraft=self.spacecraft,
                                       resample=None,
                                       pos_timestamp='center',
                                       path=self.data_path)

                self.update_viewing(viewing)
                return df, meta

        if self.spacecraft.lower() == 'soho':
            if self.sensor == 'erne':
                df, meta = soho_load(dataset="SOHO_ERNE-HED_L2-1MIN",
                                     startdate=self.start_date,
                                     enddate=self.end_date,
                                     path=self.data_path,
                                     resample=None,
                                     pos_timestamp='center')

                self.update_viewing(viewing)
                return df, meta

            if self.sensor == 'ephin':
                df, meta = soho_load(dataset="SOHO_COSTEP-EPHIN_L2-1MIN",
                                     startdate=self.start_date,
                                     enddate=self.end_date,
                                     path=self.data_path,
                                     resample=None,
                                     pos_timestamp='center')

                self.update_viewing(viewing)
                return df, meta

            if self.sensor in ("ephin-5", "ephin-15"):

                dataset = "ephin_flux_2020-2022.csv"
                datacols = ["date", "E5", "E15"]

                if os.path.isfile(f"{self.data_path}{dataset}"):
                    df = pd.read_csv(f"{self.data_path}{dataset}", usecols=datacols,
                                     index_col="date", parse_dates=True)
                else:
                    raise Warning(f"File {dataset} not found at {self.data_path}! Please verify that 'data_path' is correct.")
                meta = {"E5": "0.45 - 0.50 MeV",
                        "E15": "0.70 - 1.10 MeV"}

                self.update_viewing(viewing)
                return df, meta

        if self.spacecraft.lower() == 'wind':

            # In Wind's case we have to retrieve the original viewing before updating, because
            # otherwise viewing = 'None' will mess up everything down the road
            viewing = self.viewing

            if self.sensor == '3dp':

                df_i, meta_i = wind3dp_load(dataset="WI_SOPD_3DP",
                                            startdate=self.start_date,
                                            enddate=self.end_date,
                                            resample=None,
                                            multi_index=False,
                                            path=self.data_path,
                                            threshold=self.threshold)

                df_e, meta_e = wind3dp_load(dataset="WI_SFPD_3DP",
                                            startdate=self.start_date,
                                            enddate=self.end_date,
                                            resample=None,
                                            multi_index=False,
                                            path=self.data_path,
                                            threshold=self.threshold)

                df_omni_i, meta_omni_i = wind3dp_load(dataset="WI_SOSP_3DP",
                                                      startdate=self.start_date,
                                                      enddate=self.end_date,
                                                      resample=None,
                                                      multi_index=False,
                                                      path=self.data_path,
                                                      threshold=self.threshold)

                df_omni_e, meta_omni_e = wind3dp_load(dataset="WI_SFSP_3DP",
                                                      startdate=self.start_date,
                                                      enddate=self.end_date,
                                                      resample=None,
                                                      multi_index=False,
                                                      path=self.data_path,
                                                      threshold=self.threshold)

                self.update_viewing(viewing)
                return df_omni_i, df_omni_e, df_i, df_e, meta_i, meta_e

        if self.spacecraft.lower() == 'psp':
            if self.sensor.lower() == 'isois-epihi':
                df, meta = psp_isois_load(dataset='PSP_ISOIS-EPIHI_L2-HET-RATES60',
                                          startdate=self.start_date,
                                          enddate=self.end_date,
                                          path=self.data_path,
                                          resample=None)

                self.update_viewing(viewing)
                return df, meta
            if self.sensor.lower() == 'isois-epilo':
                df, meta = psp_isois_load(dataset='PSP_ISOIS-EPILO_L2-PE',
                                          startdate=self.start_date,
                                          enddate=self.end_date,
                                          path=self.data_path,
                                          resample=None,
                                          epilo_channel='F',
                                          epilo_threshold=self.threshold)

                self.update_viewing(viewing)
                return df, meta

        if self.spacecraft.lower() == 'bepi':
            df, meta = bepi_sixs_load(startdate=self.start_date,
                                      enddate=self.end_date,
                                      side=viewing,
                                      path=self.data_path)
            df_i = df[[f"P{i}" for i in range(1, 10)]]
            df_e = df[[f"E{i}" for i in range(1, 8)]]
            return df_i, df_e, meta

    def load_all_viewing(self):

        if self.spacecraft == 'solo':

            if self.sensor in ['het', 'ept']:

                self.df_i_sun, self.df_e_sun, self.energies_sun =\
                    self.load_data(self.spacecraft, self.sensor,
                                   'sun', self.data_level)

                self.df_i_asun, self.df_e_asun, self.energies_asun =\
                    self.load_data(self.spacecraft, self.sensor,
                                   'asun', self.data_level)

                self.df_i_north, self.df_e_north, self.energies_north =\
                    self.load_data(self.spacecraft, self.sensor,
                                   'north', self.data_level)

                self.df_i_south, self.df_e_south, self.energies_south =\
                    self.load_data(self.spacecraft, self.sensor,
                                   'south', self.data_level)

            elif self.sensor == 'step':

                self.df_step, self.energies_step =\
                    self.load_data(self.spacecraft, self.sensor, 'None',
                                   self.data_level)

        if self.spacecraft[:2].lower() == 'st':

            if self.sensor == 'sept':

                self.df_i_sun, self.df_e_sun, self.energies_i_sun, self.energies_e_sun =\
                    self.load_data(self.spacecraft, self.sensor,
                                   'sun', self.data_level)

                self.df_i_asun, self.df_e_asun, self.energies_i_asun, self.energies_e_asun =\
                    self.load_data(self.spacecraft, self.sensor,
                                   'asun', self.data_level)

                self.df_i_north, self.df_e_north, self.energies_i_north, self.energies_e_north =\
                    self.load_data(self.spacecraft, self.sensor,
                                   'north', self.data_level)

                self.df_i_south, self.df_e_south, self.energies_i_south, self.energies_e_south =\
                    self.load_data(self.spacecraft, self.sensor,
                                   'south', self.data_level)

            elif self.sensor == 'het':

                self.df_het, self.meta_het =\
                    self.load_data(self.spacecraft, self.sensor, 'None',
                                   self.data_level)
                self.current_df_i = self.df_het.filter(like='Proton')
                self.current_df_e = self.df_het.filter(like='Electron')
                self.current_energies = self.meta_het

        if self.spacecraft.lower() == 'soho':

            if self.sensor.lower() == 'erne':

                self.df, self.meta =\
                    self.load_data(self.spacecraft, self.sensor, 'None',
                                   self.data_level)
                self.current_df_i = self.df.filter(like='PH_')
                # self.current_df_e = self.df.filter(like='Electron')
                self.current_energies = self.meta

            if self.sensor.lower() == 'ephin':
                self.df, self.meta =\
                    self.load_data(self.spacecraft, self.sensor, 'None',
                                   self.data_level)
                self.current_df_e = self.df.filter(like='E')
                self.current_energies = self.meta

            if self.sensor.lower() in ("ephin-5", "ephin-15"):
                self.df, self.meta =\
                    self.load_data(self.spacecraft, self.sensor, 'None',
                                   self.data_level)
                self.current_df_e = self.df
                self.current_energies = self.meta

        if self.spacecraft.lower() == 'wind':
            if self.sensor.lower() == '3dp':
                self.df_omni_i, self.df_omni_e, self.df_i, self.df_e, self.meta_i, self.meta_e = \
                    self.load_data(self.spacecraft, self.sensor, 'None', self.data_level, threshold=self.threshold)
                # self.df_i = self.df_i.filter(like='FLUX')
                # self.df_e = self.df_e.filter(like='FLUX')
                self.current_i_energies = self.meta_i
                self.current_e_energies = self.meta_e

        if self.spacecraft.lower() == 'psp':
            if self.sensor.lower() == 'isois-epihi':
                # Note: load_data(viewing='all') doesn't really has an effect, but for PSP/ISOIS-EPIHI all viewings are always loaded anyhow.
                self.df, self.meta = self.load_data(self.spacecraft, self.sensor, 'all', self.data_level)
                self.df_e = self.df.filter(like='Electrons_Rate_')
                self.current_e_energies = self.meta
                self.df_i = self.df.filter(like='H_Flux_')
                self.current_i_energies = self.meta
            if self.sensor.lower() == 'isois-epilo':
                # Note: load_data(viewing='all') doesn't really has an effect, but for PSP/ISOIS-EPILO all viewings are always loaded anyhow.
                self.df, self.meta = self.load_data(self.spacecraft, self.sensor, 'all', self.data_level, threshold=self.threshold)
                self.df_e = self.df.filter(like='Electron_CountRate_')
                self.current_e_energies = self.meta
                # protons not yet included in PSP/ISOIS-EPILO dataset
                # self.df_i = self.df.filter(like='H_Flux_')
                # self.current_i_energies = self.meta

        if self.spacecraft.lower() == 'bepi':
            self.df_i_0, self.df_e_0, self.energies_0 =\
                self.load_data(self.spacecraft, self.sensor, viewing='0', data_level='None')
            self.df_i_1, self.df_e_1, self.energies_1 =\
                self.load_data(self.spacecraft, self.sensor, viewing='1', data_level='None')
            self.df_i_2, self.df_e_2, self.energies_2 =\
                self.load_data(self.spacecraft, self.sensor, viewing='2', data_level='None')
            # side 3 and 4 should not be used for SIXS, but they can be activated by uncommenting the following lines
            # self.df_i_3, self.df_e_3, self.energies_3 =\
            #     self.load_data(self.spacecraft, self.sensor, viewing='3', data_level='None')
            # self.df_i_4, self.df_e_4, self.energies_4 =\
            #     self.load_data(self.spacecraft, self.sensor, viewing='4', data_level='None')

    def choose_data(self, viewing):

        self.update_viewing(viewing)

        if self.spacecraft == 'solo':
            if viewing == 'sun':

                self.current_df_i = self.df_i_sun
                self.current_df_e = self.df_e_sun
                self.current_energies = self.energies_sun

            elif viewing == 'asun':

                self.current_df_i = self.df_i_asun
                self.current_df_e = self.df_e_asun
                self.current_energies = self.energies_asun

            elif viewing == 'north':

                self.current_df_i = self.df_i_north
                self.current_df_e = self.df_e_north
                self.current_energies = self.energies_north

            elif viewing == 'south':

                self.current_df_i = self.df_i_south
                self.current_df_e = self.df_e_south
                self.current_energies = self.energies_south

        if self.spacecraft[:2].lower() == 'st':
            if self.sensor == 'sept':
                if viewing == 'sun':

                    self.current_df_i = self.df_i_sun
                    self.current_df_e = self.df_e_sun
                    self.current_i_energies = self.energies_i_sun
                    self.current_e_energies = self.energies_e_sun

                elif viewing == 'asun':

                    self.current_df_i = self.df_i_asun
                    self.current_df_e = self.df_e_asun
                    self.current_i_energies = self.energies_i_asun
                    self.current_e_energies = self.energies_e_asun

                elif viewing == 'north':

                    self.current_df_i = self.df_i_north
                    self.current_df_e = self.df_e_north
                    self.current_i_energies = self.energies_i_north
                    self.current_e_energies = self.energies_e_north

                elif viewing == 'south':

                    self.current_df_i = self.df_i_south
                    self.current_df_e = self.df_e_south
                    self.current_i_energies = self.energies_i_south
                    self.current_e_energies = self.energies_e_south

        if self.spacecraft.lower() == 'wind':

            if self.sensor.lower() == '3dp':
                # The sectored data has a little different column names
                if self.viewing == "omnidirectional":

                    col_list_i = [col for col in self.df_omni_i.columns if "FLUX" in col]
                    col_list_e = [col for col in self.df_omni_e.columns if "FLUX" in col]
                    self.current_df_i = self.df_omni_i[col_list_i]
                    self.current_df_e = self.df_omni_e[col_list_e]

                else:

                    col_list_i = [col for col in self.df_i.columns if col.endswith(str(self.viewing)) and "FLUX" in col]
                    col_list_e = [col for col in self.df_e.columns if col.endswith(str(self.viewing)) and "FLUX" in col]
                    self.current_df_i = self.df_i[col_list_i]
                    self.current_df_e = self.df_e[col_list_e]

        if self.spacecraft.lower() == 'psp':
            if self.sensor.lower() == 'isois-epihi':
                # viewing = 'A' or 'B'
                self.current_df_e = self.df_e[self.df_e.columns[self.df_e.columns.str.startswith(viewing.upper())]]
                self.current_df_i = self.df_i[self.df_i.columns[self.df_i.columns.str.startswith(viewing.upper())]]
            if self.sensor.lower() == 'isois-epilo':
                # viewing = '0' to '7'
                self.current_df_e = self.df_e[self.df_e.columns[self.df_e.columns.str.endswith(viewing)]]

                # Probably just a temporary thing, but cut all channels without a corresponding energy range string in them to avoid problems with
                # dynamic spectrum. Magic number 12 is the amount of channels that have a corresponding energy description.
                col_list = [col for col in self.current_df_e.columns if int(col.split('_')[3][1:])<12]
                self.current_df_e = self.current_df_e[col_list]

                # protons not yet included in PSP/ISOIS-EPILO dataset
                # self.current_df_i = self.df_i[self.df_i.columns[self.df_i.columns.str.endswith(viewing)]]

        if self.spacecraft.lower() == 'bepi':
            if viewing == '0':
                self.current_df_i = self.df_i_0
                self.current_df_e = self.df_e_0
                self.current_energies = self.energies_0
            elif viewing == '1':
                self.current_df_i = self.df_i_1
                self.current_df_e = self.df_e_1
                self.current_energies = self.energies_1
            elif viewing == '2':
                self.current_df_i = self.df_i_2
                self.current_df_e = self.df_e_2
                self.current_energies = self.energies_2
            # side 3 and 4 should not be used for SIXS, but they can be activated by uncommenting the following lines
            # elif(viewing == '3'):
            #     self.current_df_i = self.df_i_3
            #     self.current_df_e = self.df_e_3
            #     self.current_energies = self.energies_3
            # elif(viewing == '4'):
            #     self.current_df_i = self.df_i_4
            #     self.current_df_e = self.df_e_4
            #     self.current_energies = self.energies_4

    def calc_av_en_flux_HET(self, df, energies, en_channel):

        """This function averages the flux of several
        energy channels of SolO/HET into a combined energy channel
        channel numbers counted from 0

        Parameters
        ----------
        df : pd.DataFrame DataFrame containing HET data
            DataFrame containing HET data
        energies : dict
            Energy dict returned from epd_loader (from Jan)
        en_channel : int or list
            energy channel or list with first and last channel to be used
        species : string
            'e', 'electrons', 'p', 'i', 'protons', 'ions'

        Returns
        -------
        pd.DataFrame
            flux_out: contains channel-averaged flux

        Raises
        ------
        Exception
            [description]
        """

        species = self.species

        try:

            if species not in ['e', 'electrons', 'p', 'protons', 'H']:

                raise ValueError("species not defined. Must by one of 'e',\
                                 'electrons', 'p', 'protons', 'H'")

        except ValueError as error:

            print(repr(error))
            raise

        if species in ['e', 'electrons']:

            en_str = energies['Electron_Bins_Text']
            bins_width = 'Electron_Bins_Width'
            flux_key = 'Electron_Flux'

        if species in ['p', 'protons', 'H']:

            en_str = energies['H_Bins_Text']
            bins_width = 'H_Bins_Width'
            flux_key = 'H_Flux'

            if flux_key not in df.keys():

                flux_key = 'H_Flux'

        if type(en_channel) == list:

            # An IndexError here is caused by invalid channel choice
            try:
                en_channel_string = en_str[en_channel[0]][0].split()[0] + ' - '\
                    + en_str[en_channel[-1]][0].split()[2] + ' ' +\
                    en_str[en_channel[-1]][0].split()[3]

            except IndexError:
                raise Exception(f"{en_channel} is an invalid channel or a combination of channels!")

            if len(en_channel) > 2:

                raise Exception('en_channel must have len 2 or less!')

            if len(en_channel) == 2:

                DE = energies[bins_width]

                for bins in np.arange(en_channel[0], en_channel[-1] + 1):

                    if bins == en_channel[0]:

                        I_all = df[flux_key].values[:, bins] * DE[bins]

                    else:

                        I_all = I_all + df[flux_key].values[:, bins] * DE[bins]

                DE_total = np.sum(DE[(en_channel[0]):(en_channel[-1] + 1)])
                flux_av_en = pd.Series(I_all/DE_total, index=df.index)
                flux_out = pd.DataFrame({'flux': flux_av_en}, index=df.index)

            else:

                en_channel = en_channel[0]
                flux_out = pd.DataFrame({'flux':
                                        df[flux_key].values[:, en_channel]},
                                        index=df.index)

        else:

            flux_out = pd.DataFrame({'flux':
                                    df[flux_key].values[:, en_channel]},
                                    index=df.index)
            en_channel_string = en_str[en_channel]

        return flux_out, en_channel_string

    def calc_av_en_flux_EPT(self, df, energies, en_channel):

        """This function averages the flux of several energy
        channels of EPT into a combined energy channel
        channel numbers counted from 0

        Parameters
        ----------
        df : pd.DataFrame DataFrame containing EPT data
            DataFrame containing EPT data
        energies : dict
            Energy dict returned from epd_loader (from Jan)
        en_channel : int or list
            energy channel number(s) to be used
        species : string
            'e', 'electrons', 'p', 'i', 'protons', 'ions'

        Returns
        -------
        pd.DataFrame
            flux_out: contains channel-averaged flux

        Raises
        ------
        Exception
            [description]
        """

        species = self.species

        try:

            if species not in ['e', 'electrons', 'p', 'i', 'protons', 'ions']:

                raise ValueError("species not defined. Must by one of 'e',"
                                 "'electrons', 'p', 'i', 'protons', 'ions'")

        except ValueError as error:
            print(repr(error))
            raise

        if species in ['e', 'electrons']:

            bins_width = 'Electron_Bins_Width'
            flux_key = 'Electron_Flux'
            en_str = energies['Electron_Bins_Text']

        if species in ['p', 'i', 'protons', 'ions']:

            bins_width = 'Ion_Bins_Width'
            flux_key = 'Ion_Flux'
            en_str = energies['Ion_Bins_Text']

            if flux_key not in df.keys():

                flux_key = 'H_Flux'

        if type(en_channel) == list:

            # An IndexError here is caused by invalid channel choice
            try:
                en_channel_string = en_str[en_channel[0]][0].split()[0] + ' - '\
                    + en_str[en_channel[-1]][0].split()[2] + ' '\
                    + en_str[en_channel[-1]][0].split()[3]

            except IndexError:
                raise Exception(f"{en_channel} is an invalid channel or a combination of channels!")

            if len(en_channel) > 2:

                raise Exception('en_channel must have len 2 or less!')

            if len(en_channel) == 2:

                DE = energies[bins_width]

                for bins in np.arange(en_channel[0], en_channel[-1]+1):

                    if bins == en_channel[0]:

                        I_all = df[flux_key].values[:, bins] * DE[bins]

                    else:

                        I_all = I_all + df[flux_key].values[:, bins] * DE[bins]

                DE_total = np.sum(DE[(en_channel[0]):(en_channel[-1]+1)])
                flux_av_en = pd.Series(I_all/DE_total, index=df.index)
                flux_out = pd.DataFrame({'flux': flux_av_en}, index=df.index)

            else:

                en_channel = en_channel[0]
                flux_out = pd.DataFrame({'flux':
                                        df[flux_key].values[:, en_channel]},
                                        index=df.index)

        else:

            flux_out = pd.DataFrame({'flux':
                                    df[flux_key].values[:, en_channel]},
                                    index=df.index)
            en_channel_string = en_str[en_channel]

        return flux_out, en_channel_string

    def resample(self, df_flux, resample_period):

        df_flux_out = df_flux.resample(resample_period, label='left').mean()
        df_flux_out.index = df_flux_out.index\
            + to_offset(pd.Timedelta(resample_period)/2)

        return df_flux_out

    def print_info(self, title, info):

        title_string = "##### >" + title + "< #####"
        print(title_string)
        print(info)
        print('#'*len(title_string) + '\n')

    def mean_value(self, tb_start, tb_end, flux_series):

        """
        This function calculates the classical mean of the background period
        which is used in the onset analysis.
        """

        # replace date_series with the resampled version
        date = flux_series.index
        background = flux_series.loc[(date >= tb_start) & (date < tb_end)]
        mean_value = np.nanmean(background)
        sigma = np.nanstd(background)

        return [mean_value, sigma]

    def onset_determination(self, ma_sigma, flux_series, cusum_window, bg_end_time):

        flux_series = flux_series[bg_end_time:]

        # assert date and the starting index of the averaging process
        date = flux_series.index
        ma = ma_sigma[0]
        sigma = ma_sigma[1]
        md = ma + self.x_sigma*sigma

        # k may get really big if sigma is large in comparison to mean
        try:

            k = (md-ma)/(np.log(md)-np.log(ma))
            k_round = round(k/sigma)

        except ValueError:

            # First ValueError I encountered was due to ma=md=2.0 -> k = "0/0"
            k_round = 1

        # choose h, the variable dictating the "hastiness" of onset alert
        if k < 1.0:

            h = 1

        else:

            h = 2

        alert = 0
        cusum = np.zeros(len(flux_series))
        norm_channel = np.zeros(len(flux_series))

        # set the onset as default to be NaT (Not a Date)
        onset_time = pd.NaT

        for i in range(1, len(cusum)):

            # normalize the observed flux
            norm_channel[i] = (flux_series[i]-ma)/sigma

            # calculate the value for ith cusum entry
            cusum[i] = max(0, norm_channel[i] - k_round + cusum[i-1])

            # check if cusum[i] is above threshold h,
            # if it is -> increment alert
            if cusum[i] > h:

                alert = alert + 1

            else:

                alert = 0

            # cusum_window(default:30) subsequent increments to alert
            # means that the onset was found
            if alert == cusum_window:

                onset_time = date[i - alert]
                break

        # ma = mu_a = background average
        # md = mu_d = background average + 2*sigma
        # k_round = integer value of k, that is the reference value to
        # poisson cumulative sum
        # h = 1 or 2,describes the hastiness of onset alert
        # onset_time = the time of the onset
        # S = the cusum function

        return [ma, md, k_round, norm_channel, cusum, onset_time]

    def onset_analysis(self, df_flux, windowstart, windowlen, windowrange, channels_dict,
                       channel='flux', cusum_window=30, yscale='log',
                       ylim=None, xlim=None):

        self.print_info("Energy channels", channels_dict)
        spacecraft = self.spacecraft.upper()
        sensor = self.sensor.upper()

        color_dict = {
            'onset_time': '#e41a1c',
            'bg_mean': '#e41a1c',
            'flux_peak': '#1a1682',
            'bg': '#de8585'
        }

        if self.spacecraft == 'solo':
            flux_series = df_flux[channel]
        if self.spacecraft[:2].lower() == 'st':
            flux_series = df_flux  # [channel]'
        if self.spacecraft.lower() == 'soho':
            flux_series = df_flux  # [channel]
        if self.spacecraft.lower() == 'wind':
            flux_series = df_flux  # [channel]
        if self.spacecraft.lower() == 'psp':
            flux_series = df_flux[channel]
        if self.spacecraft.lower() == 'bepi':
            flux_series = df_flux  # [channel]
        date = flux_series.index

        if ylim is None:

            ylim = [np.nanmin(flux_series[flux_series > 0]),
                    np.nanmax(flux_series) * 3]

        # windowrange is by default None, and then we define the start and stop with integer hours
        if windowrange is None:
            # dates for start and end of the averaging processes
            avg_start = date[0] + datetime.timedelta(hours=windowstart)
            # ending time is starting time + a given timedelta in hours
            avg_end = avg_start + datetime.timedelta(hours=windowlen)

        else:
            avg_start, avg_end = windowrange[0], windowrange[1]

        if xlim is None:

            xlim = [date[0], date[-1]]

        else:

            df_flux = df_flux[xlim[0]:xlim[-1]]

        # onset not yet found
        onset_found = False
        background_stats = self.mean_value(avg_start, avg_end, flux_series)
        onset_stats =\
            self.onset_determination(background_stats, flux_series,
                                     cusum_window, avg_end)

        if not isinstance(onset_stats[-1], pd._libs.tslibs.nattype.NaTType):

            onset_found = True

        if self.spacecraft == 'solo':
            df_flux_peak = df_flux[df_flux[channel] == df_flux[channel].max()]
        if self.spacecraft[:2].lower() == 'st':
            df_flux_peak = df_flux[df_flux == df_flux.max()]
        if self.spacecraft == 'soho':
            df_flux_peak = df_flux[df_flux == df_flux.max()]
        if self.spacecraft == 'wind':
            df_flux_peak = df_flux[df_flux == df_flux.max()]
        if self.spacecraft == 'psp':
            # df_flux_peak = df_flux[df_flux == df_flux.max()]
            df_flux_peak = df_flux[df_flux[channel] == df_flux[channel].max()]
        if self.spacecraft == 'bepi':
            df_flux_peak = df_flux[df_flux == df_flux.max()]
            # df_flux_peak = df_flux[df_flux[channel] == df_flux[channel].max()]
        self.print_info("Flux peak", df_flux_peak)
        self.print_info("Onset time", onset_stats[-1])
        self.print_info("Mean of background intensity",
                        background_stats[0])
        self.print_info("Std of background intensity",
                        background_stats[1])

        # Before starting the plot, save the original rcParam options and update to new ones
        original_rcparams = self.save_and_update_rcparams("onset_tool")

        fig, ax = plt.subplots()
        ax.plot(flux_series.index, flux_series.values, ds='steps-mid')

        # CUSUM and norm datapoints in plots.
        '''
        ax.scatter(flux_series.index, onset_stats[-3], s=1,
                   color='darkgreen', alpha=0.7, label='norm')
        ax.scatter(flux_series.index, onset_stats[-2], s=3,
                   c='maroon', label='CUSUM')
        '''

        # onset time
        if onset_found:

            # Onset time line
            ax.axvline(onset_stats[-1], linewidth=1.5,
                       color=color_dict['onset_time'], linestyle='-',
                       label="Onset time")

        # Flux peak line (first peak only, if there's multiple)
        try:
            ax.axvline(df_flux_peak.index[0], linewidth=1.5,
                       color=color_dict['flux_peak'], linestyle='-',
                       label="Peak time")

        except IndexError:
            exceptionmsg = "IndexError! Maybe you didn't adjust background_range or plot_range correctly?"
            raise Exception(exceptionmsg)

        # background mean
        ax.axhline(onset_stats[0], linewidth=2,
                   color=color_dict['bg_mean'], linestyle='--',
                   label="Mean of background")

        # background mean + 2*std
        ax.axhline(onset_stats[1], linewidth=2,
                   color=color_dict['bg_mean'], linestyle=':',
                   label=f"Mean + {str(self.x_sigma)} * std of background")

        # Background shaded area
        ax.axvspan(avg_start, avg_end, color=color_dict['bg'],
                   label="Background")

        # ax.set_xlabel("Time [HH:MM \nYYYY-mm-dd]", fontsize=16)
        ax.set_ylabel(r"Intensity [1/(cm$^{2}$ sr s MeV)]", fontsize=16)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        # figure limits and scale
        plt.ylim(ylim)
        plt.xlim(xlim[0], xlim[1])
        plt.yscale(yscale)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                   fancybox=True, shadow=False, ncol=3, fontsize=16)

        # tickmarks, their size etc...
        plt.tick_params(which='major', length=5, width=1.5, labelsize=16)
        plt.tick_params(which='minor', length=4, width=1)

        # date tick locator and formatter
        ax.xaxis_date()
        # ax.xaxis.set_major_locator(ticker.MaxNLocator(9))
        # utc_dt_format1 = DateFormatter('%H:%M \n%Y-%m-%d')
        utc_dt_format1 = DateFormatter('%H:%M\n%b %d\n%Y')
        ax.xaxis.set_major_formatter(utc_dt_format1)

        if self.species == 'e':

            s_identifier = 'electrons'

        if self.species in ['p', 'i']:

            if ((spacecraft == 'sta' and sensor == 'sept') or (spacecraft == 'solo' and sensor == 'ept')):

                s_identifier = 'ions'

            else:

                s_identifier = 'protons'

        self.print_info("Particle species", s_identifier)

        if (self.viewing_used != '' and self.viewing_used is not None):

            plt.title(f"{spacecraft}/{sensor} {channels_dict} {s_identifier}\n"
                      f"{self.averaging_used} averaging, viewing: "
                      f"{self.viewing_used.upper()}")

        else:

            plt.title(f"{spacecraft}/{sensor} {channels_dict} {s_identifier}\n"
                      f"{self.averaging_used} averaging")

        fig.set_size_inches(16, 8)

        # Onset label
        if onset_found:

            if (self.spacecraft == 'solo' or self.spacecraft == 'psp'):
                plabel = AnchoredText(f"Onset time: {str(onset_stats[-1])[:19]}\n"
                                      f"Peak flux: {df_flux_peak['flux'][0]:.2E}",
                                      prop=dict(size=13), frameon=True,
                                      loc=(4))
            # if(self.spacecraft[:2].lower() == 'st' or self.spacecraft == 'soho' or self.spacecraft == 'wind'):
            else:
                plabel = AnchoredText(f"Onset time: {str(onset_stats[-1])[:19]}\n"
                                      f"Peak flux: {df_flux_peak.values[0]:.2E}",
                                      prop=dict(size=13), frameon=True,
                                      loc=(4))

        else:

            plabel = AnchoredText("No onset found",
                                  prop=dict(size=13), frameon=True,
                                  loc=(4))

        plabel.patch.set_boxstyle("round, pad=0., rounding_size=0.2")
        plabel.patch.set_linewidth(2.0)

        # Background label
        blabel = AnchoredText(f"Background:\n{avg_start} - {avg_end}",
                              prop=dict(size=13), frameon=True,
                              loc='upper left')
        blabel.patch.set_boxstyle("round, pad=0., rounding_size=0.2")
        blabel.patch.set_linewidth(2.0)

        # Energy and species label
        '''
        eslabel = AnchoredText(f"{channels_dict} {s_identifier}",
                               prop=dict(size=13), frameon=True,
                               loc='lower left')
        eslabel.patch.set_boxstyle("round, pad=0., rounding_size=0.2")
        eslabel.patch.set_linewidth(2.0)
        '''

        ax.add_artist(plabel)
        ax.add_artist(blabel)
        # ax.add_artist(eslabel)
        plt.tight_layout()
        plt.show()

        # Finally reset matplotlib rcParams to what they were before plotting
        rcParams.update(original_rcparams)

        return flux_series, onset_stats, onset_found, df_flux_peak, df_flux_peak.index[0], fig, background_stats[0]

    def find_onset(self, viewing, bg_start=None, bg_length=None, background_range=None, resample_period=None,
                   channels=[0, 1], yscale='log', cusum_window=30, xlim=None, x_sigma=2):
        """
        This method runs Poisson-CUSUM onset analysis for the Event object.

        Parameters:
        -----------
        viewing : str
                        The viewing direction of the sensor.
        bg_start : int or float, default None
                        The start of background averaging from the start of the time series data in hours.
        bg_length : int or float, default None
                        The length of  the background averaging period in hours.
        background_range : tuple or list of datetimes with len=2, default None
                        The time range of background averaging. If defined, takes precedence over bg_start and bg_length.
        resample_period : str, default None
                        Pandas-compatible time string to average data. e.g. '10s' for 10 seconds or '2min' for 1 minutes.
        channels : int or list of 2 ints, default [0,1]
                        Index or a combination of indices to plot a channel or combination of channels.
        yscale : str, default 'log'
                        Matplotlib-compatible string for the scale of the y-axis. e.g. 'log' or 'linear'
        cusum_window : int, default 30
                        The amount of consecutive data points above the threshold before identifying an onset.
        xlim : tuple or list, default None
                        Panda-compatible datetimes or strings to assert the left and right boundary of the x-axis of the plot.
        x_sigma : int, default 2
                        The multiplier of m_d in the definition of the control parameter k in Poisson-CUSUM method.
        """

        # This check was initially transforming the 'channels' integer to a tuple of len==1, but that
        # raised a ValueError with solo/ept. However, a list of len==1 is somehow okay.
        if isinstance(channels, int):
            channels = [channels]

        if (background_range is not None) and (xlim is not None):
            # Check if background is separated from plot range by over a day, issue a warning if so, but don't
            if (background_range[0] < xlim[0] - datetime.timedelta(days=1) and background_range[0] < xlim[1] - datetime.timedelta(days=1)) or \
               (background_range[1] > xlim[0] + datetime.timedelta(days=1) and background_range[1] > xlim[1] + datetime.timedelta(days=1)):
                background_warning = "NOTICE that your background_range is separated from plot_range by over a day.\nIf this was intentional you may ignore this warning."
                warnings.warn(message=background_warning)

        if (self.spacecraft[:2].lower() == 'st' and self.sensor == 'sept') \
                or (self.spacecraft.lower() == 'psp' and self.sensor.startswith('isois')) \
                or (self.spacecraft.lower() == 'solo' and self.sensor == 'ept') \
                or (self.spacecraft.lower() == 'solo' and self.sensor == 'het') \
                or (self.spacecraft.lower() == 'wind' and self.sensor == '3dp') \
                or (self.spacecraft.lower() == 'bepi'):
            self.viewing_used = viewing
            self.choose_data(viewing)
        elif (self.spacecraft[:2].lower() == 'st' and self.sensor == 'het'):
            self.viewing_used = ''
        elif (self.spacecraft.lower() == 'soho' and self.sensor == 'erne'):
            self.viewing_used = ''
        elif (self.spacecraft.lower() == 'soho' and self.sensor in ["ephin", "ephin-5", "ephin-15"]):
            self.viewing_used = ''

        self.averaging_used = resample_period
        self.x_sigma = x_sigma

        if self.spacecraft == 'solo':

            if self.sensor == 'het':

                if self.species in ['p', 'i']:

                    df_flux, en_channel_string =\
                        self.calc_av_en_flux_HET(self.current_df_i,
                                                 self.current_energies,
                                                 channels)
                elif self.species == 'e':

                    df_flux, en_channel_string =\
                        self.calc_av_en_flux_HET(self.current_df_e,
                                                 self.current_energies,
                                                 channels)

            elif self.sensor == 'ept':

                if self.species in ['p', 'i']:

                    df_flux, en_channel_string =\
                        self.calc_av_en_flux_EPT(self.current_df_i,
                                                 self.current_energies,
                                                 channels)
                elif self.species == 'e':

                    df_flux, en_channel_string =\
                        self.calc_av_en_flux_EPT(self.current_df_e,
                                                 self.current_energies,
                                                 channels)

            else:
                invalid_sensor_msg = "Invalid sensor!"
                raise Exception(invalid_sensor_msg)

        if self.spacecraft[:2] == 'st':

            # Super ugly implementation, but easiest to just wrap both sept and het calculators
            # in try block. KeyError is caused by an invalid channel choice.
            try:

                if self.sensor == 'het':

                    if self.species in ['p', 'i']:

                        df_flux, en_channel_string =\
                            calc_av_en_flux_ST_HET(self.current_df_i,
                                                   self.current_energies['channels_dict_df_p'],
                                                   channels,
                                                   species='p')
                    elif self.species == 'e':

                        df_flux, en_channel_string =\
                            calc_av_en_flux_ST_HET(self.current_df_e,
                                                   self.current_energies['channels_dict_df_e'],
                                                   channels,
                                                   species='e')

                elif self.sensor == 'sept':

                    if self.species in ['p', 'i']:

                        df_flux, en_channel_string =\
                            calc_av_en_flux_SEPT(self.current_df_i,
                                                 self.current_i_energies,
                                                 channels)
                    elif self.species == 'e':

                        df_flux, en_channel_string =\
                            calc_av_en_flux_SEPT(self.current_df_e,
                                                 self.current_e_energies,
                                                 channels)

            except KeyError:
                raise Exception(f"{channels} is an invalid channel or a combination of channels!")

        if self.spacecraft == 'soho':

            # A KeyError here is caused by invalid channel
            try:

                if self.sensor == 'erne':

                    if self.species in ['p', 'i']:

                        df_flux, en_channel_string =\
                            calc_av_en_flux_ERNE(self.current_df_i,
                                                 self.current_energies['channels_dict_df_p'],
                                                 channels,
                                                 species='p',
                                                 sensor='HET')

                if self.sensor == 'ephin':
                    # convert single-element "channels" list to integer
                    if type(channels) == list:
                        if len(channels) == 1:
                            channels = channels[0]
                        else:
                            print("No multi-channel support for SOHO/EPHIN included yet! Select only one single channel.")
                    if self.species == 'e':
                        df_flux = self.current_df_e[f'E{channels}']
                        en_channel_string = self.current_energies[f'E{channels}']

                if self.sensor in ("ephin-5", "ephin-15"):
                    if isinstance(channels, list):
                        if len(channels) == 1:
                            channels = channels[0]
                        else:
                            raise Exception("No multi-channel support for SOHO/EPHIN included yet! Select only one single channel.")
                    if self.species == 'e':
                        df_flux = self.current_df_e[f"E{channels}"]
                        en_channel_string = self.current_energies[f"E{channels}"]

            except KeyError:
                raise Exception(f"{channels} is an invalid channel or a combination of channels!")

        if self.spacecraft == 'wind':
            if self.sensor == '3dp':
                # convert single-element "channels" list to integer
                if type(channels) == list:
                    if len(channels) == 1:
                        channels = channels[0]
                    else:
                        print("No multi-channel support for Wind/3DP included yet! Select only one single channel.")
                if self.species in ['p', 'i']:
                    if viewing != "omnidirectional":
                        df_flux = self.current_df_i.filter(like=f'FLUX_E{channels}')
                    else:
                        df_flux = self.current_df_i.filter(like=f'FLUX_{channels}')
                    # extract pd.Series for further use:
                    df_flux = df_flux[df_flux.columns[0]]
                    # change flux units from '#/cm2-ster-eV-sec' to '#/cm2-ster-MeV-sec'
                    df_flux = df_flux*1e6
                    en_channel_string = self.current_i_energies['channels_dict_df']['Bins_Text'][f'ENERGY_{channels}']
                elif self.species == 'e':
                    if viewing != "omnidirectional":
                        df_flux = self.current_df_e.filter(like=f'FLUX_E{channels}')
                    else:
                        df_flux = self.current_df_e.filter(like=f'FLUX_{channels}')
                    # extract pd.Series for further use:
                    df_flux = df_flux[df_flux.columns[0]]
                    # change flux units from '#/cm2-ster-eV-sec' to '#/cm2-ster-MeV-sec'
                    df_flux = df_flux*1e6
                    en_channel_string = self.current_e_energies['channels_dict_df']['Bins_Text'][f'ENERGY_{channels}']

        if self.spacecraft.lower() == 'bepi':
            if type(channels) == list:
                if len(channels) == 1:
                    # convert single-element "channels" list to integer
                    channels = channels[0]
                    if self.species == 'e':
                        df_flux = self.current_df_e[f'E{channels}']
                        en_channel_string = self.current_energies['Energy_Bin_str'][f'E{channels}']
                    if self.species in ['p', 'i']:
                        df_flux = self.current_df_i[f'P{channels}']
                        en_channel_string = self.current_energies['Energy_Bin_str'][f'P{channels}']
                else:
                    if self.species == 'e':
                        df_flux, en_channel_string = calc_av_en_flux_sixs(self.current_df_e, channels, self.species)
                    if self.species in ['p', 'i']:
                        df_flux, en_channel_string = calc_av_en_flux_sixs(self.current_df_i, channels, self.species)

        if self.spacecraft.lower() == 'psp':
            if self.sensor.lower() == 'isois-epihi':
                if self.species in ['p', 'i']:
                    # We're using here only the HET instrument of EPIHI (and not LET1 or LET2)
                    df_flux, en_channel_string =\
                        calc_av_en_flux_PSP_EPIHI(df=self.current_df_i,
                                                  energies=self.current_i_energies,
                                                  en_channel=channels,
                                                  species='p',
                                                  instrument='het',
                                                  viewing=viewing.upper())
                if self.species == 'e':
                    # We're using here only the HET instrument of EPIHI (and not LET1 or LET2)
                    df_flux, en_channel_string =\
                        calc_av_en_flux_PSP_EPIHI(df=self.current_df_e,
                                                  energies=self.current_e_energies,
                                                  en_channel=channels,
                                                  species='e',
                                                  instrument='het',
                                                  viewing=viewing.upper())
            if self.sensor.lower() == 'isois-epilo':
                if self.species == 'e':
                    # We're using here only the F channel of EPILO (and not E or G)
                    df_flux, en_channel_string =\
                        calc_av_en_flux_PSP_EPILO(df=self.current_df_e,
                                                  en_dict=self.current_e_energies,
                                                  en_channel=channels,
                                                  species='e',
                                                  mode='pe',
                                                  chan='F',
                                                  viewing=viewing)

        if resample_period is not None:

            df_averaged = self.resample(df_flux, resample_period)

        else:

            df_averaged = df_flux

        flux_series, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean =\
            self.onset_analysis(df_averaged, bg_start, bg_length, background_range,
                                en_channel_string, yscale=yscale, cusum_window=cusum_window, xlim=xlim)

        # At least in the case of solo/ept the peak_flux is a pandas Dataframe, but it should be a Series
        if isinstance(peak_flux, pd.core.frame.DataFrame):
            peak_flux = pd.Series(data=peak_flux.values[0])

        # update class attributes before returning variables:
        self.update_onset_attributes(flux_series, onset_stats, onset_found, peak_flux.values[0], peak_time, fig, bg_mean)

        return flux_series, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean

    # For backwards compatibility, make a copy of the `find_onset` function that is called `analyse` (which was its old name).
    analyse = copy.copy(find_onset)

    def dynamic_spectrum(self, view, cmap: str = 'magma', xlim: tuple = None, resample: str = None, save: bool = False) -> None:
        """
        Shows all the different energy channels in a single 2D plot, and color codes the corresponding intensity*energy^2 by a colormap.

        Parameters:
        -----------
        view : str or None
                The viewing direction of the sensor
        cmap : str, default='magma'
                The colormap for the dynamic spectrum plot
        xlim : 2-tuple of datetime strings (str, str)
                Pandas-compatible datetime strings for the start and stop of the figure
        resample : str
                Pandas-compatibe resampling string, e.g. '10min' or '30s'
        save : bool
                Saves the image
        """

        # Event attributes
        spacecraft = self.spacecraft.lower()
        instrument = self.sensor.lower()
        species = self.species

        self.choose_data(view)

        if self.spacecraft == "solo":
            if species in ("electron", 'e'):
                particle_data = self.current_df_e["Electron_Flux"]
                s_identifier = "electrons"
            else:
                try:
                    particle_data = self.current_df_i["Ion_Flux"]
                    s_identifier = "ions"
                except KeyError:
                    particle_data = self.current_df_i["H_Flux"]
                    s_identifier = "protons"

        if self.spacecraft[:2] == "st":
            if species in ["electron", 'e']:
                if instrument == "sept":
                    particle_data = self.current_df_e[[ch for ch in self.current_df_e.columns if ch[:2] == "ch"]]
                else:
                    particle_data = self.current_df_e[[ch for ch in self.current_df_e.columns if "Flux" in ch]]
                s_identifier = "electrons"
            else:
                if instrument == "sept":
                    particle_data = self.current_df_i[[ch for ch in self.current_df_i.columns if ch[:2] == "ch"]]
                    s_identifier = "ions"
                else:
                    particle_data = self.current_df_i[[ch for ch in self.current_df_i.columns if "Flux" in ch]]
                s_identifier = "protons"

        if self.spacecraft == "soho":
            if instrument.lower() == "erne":
                particle_data = self.current_df_i
                s_identifier = "protons"
            if instrument.lower() == "ephin":
                particle_data = self.current_df_e
                s_identifier = "electrons"
                warnings.warn('SOHO/EPHIN data is not fully implemented yet!')

        if spacecraft == "psp":
            if instrument.lower() == "isois-epihi":
                if species in ("electron", 'e'):
                    particle_data = self.current_df_e
                    s_identifier = "electrons"
                if species in ("proton", "p"):
                    particle_data = self.current_df_i
                    s_identifier = "protons"
            if instrument.lower() == "isois-epilo":
                if species in ("electron", 'e'):
                    particle_data = self.current_df_e
                    s_identifier = "electrons"

        if spacecraft == "wind":
            if instrument.lower() == "3dp":
                if species in ("electron", 'e'):
                    # change flux units from '#/cm2-ster-eV-sec' to '#/cm2-ster-MeV-sec'
                    particle_data = self.current_df_e*1e6
                    s_identifier = "electrons"
                else:
                    # change flux units from '#/cm2-ster-eV-sec' to '#/cm2-ster-MeV-sec'
                    particle_data = self.current_df_i*1e6
                    s_identifier = "protons"

        # These particle instruments will have keVs on their y-axis
        LOW_ENERGY_SENSORS = ("sept", "ept")

        if instrument in LOW_ENERGY_SENSORS:
            y_multiplier = 1e-3  # keV
            y_unit = "keV"
        else:
            y_multiplier = 1e-6  # MeV
            y_unit = "MeV"

        # Resample only if requested
        if resample is not None:
            particle_data = particle_data.resample(resample).mean()

        if xlim is None:
            df = particle_data[:]
            t_start, t_end = df.index[0], df.index[-1]
        else:
            # td is added to the end to avert white pixels at the end of the plot
            td_str = resample if resample is not None else '0s'
            td = pd.Timedelta(value=td_str)
            t_start, t_end = pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1])
            df = particle_data.loc[(particle_data.index >= t_start) & (particle_data.index <= (t_end+td))]

        # In practice this seeks the date on which the highest flux is observed
        date_of_event = df.iloc[np.argmax(df[df.columns[0]])].name.date()

        # Assert time and channel bins
        time = df.index

        # The low and high ends of each energy channel
        e_lows, e_highs = self.get_channel_energy_values()  # this function return energy in eVs

        # The mean energy of each channel in eVs
        mean_energies = np.sqrt(np.multiply(e_lows, e_highs))

        # Energy boundaries of plotted bins in keVs are the y-axis:
        y_arr = np.append(e_lows, e_highs[-1]) * y_multiplier

        # Set image pixel length and height
        image_len = len(time)
        image_hei = len(y_arr)-1

        # Init the grid
        grid = np.zeros((image_len, image_hei))

        # Display energy in MeVs -> multiplier squared is 1e-6*1e-6 = 1e-12
        energy_multiplier_squared = 1e-12

        # Assign grid bins -> intensity * energy^2
        for i, channel in enumerate(df):

            grid[:, i] = df[channel]*(mean_energies[i]*mean_energies[i]*energy_multiplier_squared)  # Intensity*Energy^2, and energy is in eV -> tranform to keV or MeV

        # Finally cut the last entry and transpose the grid so that it can be plotted correctly
        grid = grid[:-1, :]
        grid = grid.T

        maskedgrid = np.where(grid == 0, 0, 1)
        maskedgrid = np.ma.masked_where(maskedgrid == 1, maskedgrid)

        # ---only plotting_commands from this point----->

        # Save the original rcParams and update to new ones
        original_rcparams = self.save_and_update_rcparams("dynamic_spectrum")

        normscale = cl.LogNorm()

        # Init the figure and axes
        if self.radio_spacecraft is None:
            figsize = (27, 14)
            fig, ax = plt.subplots(figsize=figsize)
            ax = np.array([ax])
            DYN_SPEC_INDX = 0

        else:
            figsize = (27, 18)
            fig, ax = plt.subplots(nrows=2, figsize=figsize, sharex=True)
            DYN_SPEC_INDX = 1

            from seppy.tools.swaves import plot_swaves
            ax[0], colormesh = plot_swaves(downloaded_files=self.radio_files, spacecraft=self.radio_spacecraft[0], start_time=t_start, end_time=t_end, ax=ax[0], cmap=cmap)

            fig.tight_layout(pad=9.5, w_pad=-0.5, h_pad=1.0)
            # plt.subplots_adjust(wspace=-1, hspace=-1.8)

            # Colorbar
            cb = fig.colorbar(colormesh, orientation='vertical', ax=ax[0])
            clabel = "Intensity" + "\n" + "[dB]"
            cb.set_label(clabel)

        # Colormesh
        cplot = ax[DYN_SPEC_INDX].pcolormesh(time, y_arr, grid, shading='auto', cmap=cmap, norm=normscale)
        greymesh = ax[DYN_SPEC_INDX].pcolormesh(time, y_arr, maskedgrid, shading='auto', cmap='Greys', vmin=-1, vmax=1)

        # Colorbar
        cb = fig.colorbar(cplot, orientation='vertical', ax=ax[DYN_SPEC_INDX])
        clabel = r"Intensity $\cdot$ $E^{2}$" + "\n" + r"[MeV/(cm$^{2}$ sr s)]"
        cb.set_label(clabel)

        # y-axis settings
        ax[DYN_SPEC_INDX].set_yscale('log')
        ax[DYN_SPEC_INDX].set_ylim(np.nanmin(y_arr), np.nanmax(y_arr))
        ax[DYN_SPEC_INDX].set_yticks([yval for yval in y_arr])
        ax[DYN_SPEC_INDX].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        # gets rid of minor ticks and labels
        ax[DYN_SPEC_INDX].yaxis.set_tick_params(length=0, width=0, which='minor', labelsize=0.)
        ax[DYN_SPEC_INDX].yaxis.set_tick_params(length=9., width=1.5, which='major')

        ax[DYN_SPEC_INDX].set_ylabel(f"Energy [{y_unit}]")

        # x-axis settings
        # ax[DYN_SPEC_INDX].set_xlabel("Time [HH:MM \nm-d]")
        ax[DYN_SPEC_INDX].xaxis_date()
        ax[DYN_SPEC_INDX].set_xlim(t_start, t_end)
        # ax[DYN_SPEC_INDX].xaxis.set_major_locator(mdates.HourLocator(interval = 1))
        # utc_dt_format1 = DateFormatter('%H:%M \n%m-%d')
        utc_dt_format1 = DateFormatter('%H:%M\n%b %d\n%Y')
        ax[DYN_SPEC_INDX].xaxis.set_major_formatter(utc_dt_format1)
        # ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 5))

        # Title
        if view is not None:
            title = f"{spacecraft.upper()}/{instrument.upper()} ({view}) {s_identifier}, {date_of_event}"
        else:
            title = f"{spacecraft.upper()}/{instrument.upper()} {s_identifier}, {date_of_event}"

        if self.radio_spacecraft is None:
            ax[0].set_title(title)
        else:
            ax[0].set_title(f"Radio & Dynamic Spectrum, {title}")

        # saving of the figure
        if save:
            plt.savefig(f'plots/{spacecraft}_{instrument}_{date_of_event}_dynamic_spectra.png', transparent=False,
                        facecolor='white', bbox_inches='tight')

        self.fig = fig
        plt.show()

        # Finally return plotting options to what they were before plotting
        rcParams.update(original_rcparams)

    def tsa_plot(self, view, selection=None, xlim=None, resample=None):
        """
        Makes an interactive time-shift plot

        Parameters:
        ----------
        view : str or None
                    Viewing direction for the chosen sensor
        selection : 2-tuple
                    The indices of the channels one wishes to plot. End-exclusive.
        xlim : 2-tuple
                    The start and end point of the plot as pandas-compatible datetimes or strings
        resample : str
                    Pandas-compatible resampling time-string, e.g. "2min" or "50s"
        """

        import ipywidgets as widgets

        # inits
        spacecraft = self.spacecraft
        instrument = self.sensor
        species = self.species

        # This here is an extremely stupid thing, but we must convert spacecraft input name back
        # to its original version so that sunpy.get_horizon_coords() understands it
        if spacecraft == "solo":
            spacecraft_input_name = "Solar Orbiter"
        elif spacecraft == "sta":
            spacecraft_input_name = "STEREO-A"
        elif spacecraft == "stb":
            spacecraft_input_name = "STEREO-B"
        else:
            spacecraft_input_name = spacecraft.upper()

        # get (lon, lat, radius) in (deg, deg, AU) in Stonyhurst coordinates:
        # e.g. 'Solar Orbiter', 'STEREO-A', 'STEREO-B', 'SOHO', 'PSP'
        position = get_horizons_coord(spacecraft_input_name, self.start_date)
        radial_distance_value = np.round(position.radius.value, 2)

        METERS_PER_AU = 1 * u.AU.to(u.m)

        self.choose_data(view)

        if self.spacecraft == "solo":
            if species in ["electron", 'e']:
                particle_data = self.current_df_e["Electron_Flux"]
                s_identifier = "electrons"
            else:
                try:
                    particle_data = self.current_df_i["Ion_Flux"]
                    s_identifier = "ions"
                except KeyError:
                    particle_data = self.current_df_i["H_Flux"]
                    s_identifier = "protons"
            sc_identifier = "Solar Orbiter"

        if self.spacecraft[:2] == "st":
            if species in ["electron", 'e']:
                if instrument == "sept":
                    particle_data = self.current_df_e[[ch for ch in self.current_df_e.columns if ch[:2] == "ch"]]
                else:
                    particle_data = self.current_df_e[[ch for ch in self.current_df_e.columns if "Flux" in ch]]
                s_identifier = "electrons"
            else:
                if instrument == "sept":
                    particle_data = self.current_df_i[[ch for ch in self.current_df_i.columns if ch[:2] == "ch"]]
                else:
                    particle_data = self.current_df_i[[ch for ch in self.current_df_i.columns if "Flux" in ch]]
                s_identifier = "protons"
            sc_identifier = "STEREO-A" if spacecraft[-1] == "a" else "STEREO-B"

        if self.spacecraft == "soho":
            # ERNE-HED (only protons)
            if instrument.lower() == "erne":
                particle_data = self.current_df_i
                s_identifier = "protons"
            # EPHIN, as of now only electrons, could be extended to protons in the future
            if instrument.lower() == "ephin":
                particle_data = self.current_df_e
                s_identifier = "electrons"
            sc_identifier = spacecraft.upper()

        if self.spacecraft == "psp":
            if instrument.lower() == "isois-epihi":
                if species in ("electron", 'e'):
                    particle_data = self.current_df_e
                    s_identifier = "electrons"
                if species in ("proton", 'p'):
                    particle_data = self.current_df_i
                    s_identifier = "protons"

            # EPILO only has electrons
            if instrument.lower() == "isois-epilo":
                if species in ("electron", 'e'):
                    particle_data = self.current_df_e
                    s_identifier = "electrons"
            sc_identifier = "Parker Solar Probe"

        if spacecraft == "wind":
            if instrument.lower() == "3dp":
                if species in ("electron", 'e'):
                    # change flux units from '#/cm2-ster-eV-sec' to '#/cm2-ster-MeV-sec'
                    particle_data = self.current_df_e*1e6
                    s_identifier = "electrons"
                else:
                    # change flux units from '#/cm2-ster-eV-sec' to '#/cm2-ster-MeV-sec'
                    particle_data = self.current_df_i*1e6
                    s_identifier = "protons"
            sc_identifier = spacecraft.capitalize()

        # make a copy to make sure original data is not altered
        dataframe = particle_data.copy()

        particle_speeds = self.calculate_particle_speeds()

        # t_0 = t - L/v -> L/v is the coefficient that shifts the x-axis
        shift_coefficients = [METERS_PER_AU/v for v in particle_speeds]

        stepsize = 0.05
        min_slider_val, max_slider_val = 0.0, 2.55

        # Only the selected channels will be plotted
        if selection is not None:

            # len==3 means that we only choose every selection[2]:th channel
            if len(selection) == 3:
                channel_indices = [i for i in range(selection[0], selection[1], selection[2])]
                selected_channels = [channel for channel in dataframe.columns[channel_indices]]
            else:
                channel_indices = [i for i in range(selection[0], selection[1])]
                selected_channels = dataframe.columns[selection[0]:selection[1]]
        else:
            selected_channels = dataframe.columns
            channel_indices = [i for i in range(len(selected_channels))]

        # Change 0-values to nan purely for plotting purposes, since we're not doing any
        # calculations with them
        dataframe[dataframe[selected_channels] == 0] = np.nan

        # Get the channel numbers (not the indices!)
        if instrument != "isois-epilo":
            try:
                channel_nums = [int(name.split('_')[-1]) for name in selected_channels]

            except ValueError:

                # In the case of Wind/3DP, channel strings are like: FLUX_E0_P0, E for energy channel and P for direction
                if self.spacecraft == "wind":
                    channel_nums = [int(name.split('_')[1][-1]) for name in selected_channels]

                # SOHO/EPHIN has channels such as E300 etc...
                if self.spacecraft == "soho":
                    channel_nums = [name for name in selected_channels]
        else:
            channel_nums = [int(name.split('_E')[-1].split('_')[0]) for name in selected_channels]

        # Channel energy values as strings:
        channel_energy_strs = self.get_channel_energy_values("str")
        if selection is not None:
            if len(selection) == 3:
                channel_energy_strs = channel_energy_strs[slice(selection[0], selection[1], selection[2])]
            else:
                channel_energy_strs = channel_energy_strs[slice(selection[0], selection[1])]

        # creation of the figure
        fig, ax = plt.subplots(figsize=(9, 6))

        # settings of title
        ax.set_title(f"{sc_identifier} {instrument.upper()}, {s_identifier}")

        ax.grid(True)

        # settings for y and x axes
        ax.set_yscale("log")
        ax.set_ylabel(r"Intensity [1/(cm$^{2}$ sr s MeV)]")

        ax.set_xlabel(r"$t_{0} = t - L/v$")
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M\n%b %d'))

        if xlim is None:
            ax.set_xlim(dataframe.index[0], dataframe.index[-1])
        else:
            try:
                ax.set_xlim(xlim[0], xlim[1])
            except ValueError:
                ax.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))

        # So far I'm not sure how to return the original rcParams back to what they were in the case of this function,
        # because it runs interactively and there is no clear "ending" point to the function.
        # For now I'll attach the original rcparams to a class attribute, so that the user may manually return the parameters
        # after they are done with tsa.
        self.original_rcparams = self.save_and_update_rcparams("tsa")

        # housekeeping lists
        series_natural = []
        series_norm = []
        plotted_natural = []
        plotted_norm = []

        # go through the selected channels to create individual series and plot them
        for i, channel in enumerate(selected_channels):

            # construct series and its normalized counterpart
            series = flux2series(dataframe[channel], dataframe.index, resample)
            series_normalized = flux2series(series.values/np.nanmax(series.values), series.index, resample)

            # store all series to arrays for later referencing
            series_natural.append(series)
            series_norm.append(series_normalized)

            # save the plotted lines, NOTICE that they come inside a list of len==1
            p1 = ax.step(series.index, series.values, c=f"C{i}", visible=True, label=f"{channel_nums[i]}: {channel_energy_strs[i]}")
            p2 = ax.step(series_normalized.index, series_normalized.values, c=f"C{i}", visible=False)  # normalized lines are initially hidden

            # store plotted line objects for later referencing
            plotted_natural.append(p1[0])
            plotted_norm.append(p2[0])

        plt.legend(loc=1, bbox_to_anchor=(1.0, 0.25), fancybox=True, shadow=False, ncol=1, fontsize=9)

        # widget objects, slider and button
        style = {'description_width': 'initial'}
        slider = widgets.FloatSlider(value=min_slider_val,
                                     min=min_slider_val,
                                     max=max_slider_val,
                                     step=stepsize,
                                     continuous_update=True,
                                     description="Path length L [AU]: ",
                                     style=style
                                     )

        # button = widgets.Checkbox(value = False,
        #                           description = "Normalize",
        #                           indent = True
        #                           )
        button = widgets.RadioButtons(value='original data',
                                      description='Intensity:',
                                      options=['original data', 'normalized'],
                                      disabled=False
                                      )

        # A box for the path length
        path_label = f"R={radial_distance_value:.2f} AU\nL = {slider.value} AU"
        text = plt.text(0.02, 0.03, path_label, transform=ax.transAxes,
                        bbox=dict(boxstyle="square",
                                  ec=(0., 0., 0.),
                                  fc=(1., 1.0, 1.0),
                                  ))

        def timeshift(sliderobject):
            """
            timeshift connects the slider to the shifting of the plotted curves
            """
            # shift the x-values (times) by the timedelta
            for i, line in enumerate(plotted_natural):

                # calculate the timedelta in seconds corresponding to the change in the path length
                # The relevant part here is sliderobject["old"]! It saves the previous value of the slider!
                timedelta_sec = shift_coefficients[i]*(slider.value - sliderobject["old"])

                # Update the time value
                line.set_xdata(line.get_xdata() - pd.Timedelta(seconds=timedelta_sec))

            for i, line in enumerate(plotted_norm):

                # calculate the timedelta in seconds corresponding to the change in the path length
                # The relevant part here is sliderobject["old"]! It saves the previous value of the slider!
                timedelta_sec = shift_coefficients[i]*(slider.value - sliderobject["old"])

                # Update the time value
                line.set_xdata(line.get_xdata() - pd.Timedelta(seconds=timedelta_sec))

            # Update the path label artist
            text.set_text(f"R={radial_distance_value:.2f} AU\nL = {slider.value} AU")

            # Effectively this refreshes the figure
            fig.canvas.draw_idle()

        def normalize_axes(button):
            """
            this function connects the button to switching visibility of natural / normed curves
            """
            # flip the truth values of natural and normed intensity visibility
            for line in plotted_natural:
                line.set_visible(not line.get_visible())

            for line in plotted_norm:
                line.set_visible(not line.get_visible())

            # Reset the y-axis label
            if plotted_natural[0].get_visible():
                ax.set_ylabel(r"Intensity [1/(cm$^{2}$ sr s MeV)]")
            else:
                ax.set_ylabel("Intensity normalized")

            # Effectively this refreshes the figure
            fig.canvas.draw_idle()

        slider.observe(timeshift, names="value")
        display(slider)

        button.observe(normalize_axes)
        display(button)

    def get_channel_energy_values(self, returns: str = "num") -> list:
        """
        A class method to return the energies of each energy channel in either str or numerical form.

        Parameters:
        -----------
        returns: str, either 'str' or 'num'

        Returns:
        ---------
        energy_ranges : list of energy ranges as strings
        or
        lower_bounds : list of lower bounds of each energy channel in eVs
        higher_bounds : list of higher bounds of each energy channel in eVs
        """

        # First check by spacecraft, then by sensor
        if self.spacecraft == "solo":

            # All solo energies are in the same object
            energy_dict = self.current_energies

            if self.species == 'e':
                energy_ranges = energy_dict["Electron_Bins_Text"]
            else:
                try:
                    energy_ranges = energy_dict["Ion_Bins_Text"]
                except KeyError:
                    energy_ranges = energy_dict["H_Bins_Text"]

            # Each element in the list is also a list with len==1, so fix that
            energy_ranges = [element[0] for element in energy_ranges]

        if self.spacecraft[:2] == "st":

            # STEREO/SEPT energies come in two different objects
            if self.sensor == "sept":
                if self.species == 'e':
                    energy_df = self.current_e_energies
                else:
                    energy_df = self.current_i_energies

                energy_ranges = energy_df["ch_strings"].values

            # STEREO/HET energies all in the same dictionary
            else:
                energy_dict = self.current_energies

                if self.species == 'e':
                    energy_ranges = energy_dict["Electron_Bins_Text"]
                else:
                    energy_ranges = energy_dict["Proton_Bins_Text"]

                # Each element in the list is also a list with len==1, so fix that
                energy_ranges = [element[0] for element in energy_ranges]

        if self.spacecraft == "soho":
            if self.sensor.lower() == "erne":
                energy_ranges = self.current_energies["channels_dict_df_p"]["ch_strings"].values
            if self.sensor.lower() == "ephin":
                # Choose only the 4 first channels / descriptions, since I only know of
                # E150, E300, E1300 and E3000. The rest are unknown to me.
                # Go up to index 5, because index 1 is 'deactivated bc. of failure mode D'
                energy_ranges = [val for val in self.current_energies.values()][:5]
            if self.sensor.lower() in ("ephin-5", "ephin-15"):
                energy_ranges = [value for key, value in self.current_energies.items()]

        if self.spacecraft == "psp":
            energy_dict = self.meta

            if self.sensor == "isois-epihi":
                if self.species == 'e':
                    energy_ranges = energy_dict["Electrons_ENERGY_LABL"]
                if self.species == 'p':
                    energy_ranges = energy_dict["H_ENERGY_LABL"]

                # In the case of ISOIS-EPIHI, each iterable object is a list with len=1 that contains
                # the str
                energy_ranges = [element[0] for element in energy_ranges]

            if self.sensor == "isois-epilo":
                # The metadata of ISOIS-EPILO comes in a bit of complex form, so some handling is required
                if self.species == 'e':
                    chan = 'F'

                    energies = self.meta[f"Electron_Chan{chan}_Energy"].filter(like=f"_P{self.viewing}").values

                    # Calculate low and high boundaries from mean energy and energy deltas
                    energies_low = energies - self.meta[f"Electron_Chan{chan}_Energy_DELTAMINUS"].filter(like=f"_P{self.viewing}").values
                    energies_high = energies + self.meta[f"Electron_Chan{chan}_Energy_DELTAPLUS"].filter(like=f"_P{self.viewing}").values

                    # Round the numbers to one decimal place
                    energies_low_rounded = np.round(energies_low, 1)
                    energies_high_rounded = np.round(energies_high, 1)

                    # I think nan values should be removed at this point. However, if we were to do that, then print_energies()
                    # will not work anymore since tha number of channels and channel energy ranges won't be the same.
                    # In the current state PSP/ISOIS-EPILO cannot be examined with dynamic_spectrum(), because there are nan values
                    # in the channel energy ranges.
                    # energies_low_rounded = np.array([val for val in energies_low_rounded if not np.isnan(val)])
                    # energies_high_rounded = np.array([val for val in energies_high_rounded if not np.isnan(val)])

                    # produce energy range strings from low and high boundaries
                    energy_ranges = np.array([str(energies_low_rounded[i]) + ' - ' + str(energies_high_rounded[i]) + " keV" for i in range(len(energies_low_rounded))])

                    # Probably just a temporary thing, but cut all the range strings containing ´nan´ in them to avoid problems with
                    # dynamic spectrum
                    energy_ranges = np.array([s for s in energy_ranges if "nan" not in s])

        if self.spacecraft == "wind":

            if self.species == 'e':
                energy_ranges = np.array(self.meta_e["channels_dict_df"]["Bins_Text"])
            if self.species == 'p':
                energy_ranges = np.array(self.meta_i["channels_dict_df"]["Bins_Text"])

        # Check what to return before running calculations
        if returns == "str":
            return energy_ranges

        # From this line onward we extract the numerical values from low and high boundaries, and return floats, not strings
        lower_bounds, higher_bounds = [], []
        for energy_str in energy_ranges:

            # Sometimes there is no hyphen, but then it's not a range of energies
            try:
                lower_bound, temp = energy_str.split('-')
            except ValueError:
                continue

            # Generalize a bit here, since temp.split(' ') may yield a variety of different lists
            components = temp.split(' ')
            try:

                # PSP meta strings can have up to 4 spaces
                if self.spacecraft == "psp":
                    higher_bound, energy_unit = components[-2], components[-1]

                # SOHO/ERNE meta string has space, high value, space, energy_str
                elif self.spacecraft == "soho" and self.sensor == "erne":
                    higher_bound, energy_unit = components[1], components[-1]

                # Normal meta strs have two components: bounds and the energy unit
                else:
                    higher_bound, energy_unit = components

            # It could be that the strings are not in a standard format, so check if
            # there is an empty space before the second energy value
            except ValueError:

                try:
                    _, higher_bound, energy_unit = components

                # It could even be that for some godforsaken reason there are empty spaces
                # between the numbers themselves, so take care of that too
                except ValueError:

                    if components[-1] not in ["keV", "MeV"]:
                        higher_bound, energy_unit = components[1], components[2]
                    else:
                        higher_bound, energy_unit = components[1]+components[2], components[-1]

            lower_bounds.append(float(lower_bound))
            higher_bounds.append(float(higher_bound))

        # Transform lists to numpy arrays for performance and convenience
        lower_bounds, higher_bounds = np.asarray(lower_bounds), np.asarray(higher_bounds)

        # Finally before returning the lists, make sure that the unit of energy is eV
        if energy_unit == "keV":
            lower_bounds, higher_bounds = lower_bounds * 1e3, higher_bounds * 1e3

        elif energy_unit == "MeV":
            lower_bounds, higher_bounds = lower_bounds * 1e6, higher_bounds * 1e6

        # This only happens with ephin, which has MeV as the unit of energy
        else:
            lower_bounds, higher_bounds = lower_bounds * 1e6, higher_bounds * 1e6

        return lower_bounds, higher_bounds

    def calculate_particle_speeds(self):
        """
        Calculates average particle speeds by input channel energy boundaries.
        """

        if self.species in ["electron", 'e']:
            m_species = const.m_e.value
        if self.species in ['p', "ion", 'H']:
            m_species = const.m_p.value

        C_SQUARED = const.c.value*const.c.value

        # E=mc^2, a fundamental property of any object with mass
        mass_energy = m_species*C_SQUARED  # e.g. 511 keV/c for electrons

        # Get the energies of each energy channel, to calculate the mean energy of particles and ultimately
        # To get the dimensionless speeds of the particles (beta)
        e_lows, e_highs = self.get_channel_energy_values()  # get_energy_channels() returns energy in eVs

        mean_energies = np.sqrt(np.multiply(e_lows, e_highs))

        # Transform kinetic energy from electron volts to joules
        e_Joule = [((En*u.eV).to(u.J)).value for En in mean_energies]

        # Beta, the unitless speed (v/c)
        beta = [np.sqrt(1-((e_J/mass_energy + 1)**(-2))) for e_J in e_Joule]

        return np.array(beta)*const.c.value

    def print_energies(self):
        """
        Prints out the channel name / energy range pairs
        """

        # This has to be run first, otherwise self.current_df does not exist
        # Note that PSP will by default have its viewing=="all", which does not yield proper dataframes
        if self.viewing != "all":
            self.choose_data(self.viewing)
        else:
            if self.sensor == "isois-epihi":
                # Just choose data with either ´A´ or ´B´. I'm not sure if there's a difference
                self.choose_data('A')
            if self.sensor == "isois-epilo":
                # ...And here with ´3´ or ´7´. Again I don't know if it'll make a difference
                self.choose_data('3')

        if self.species in ['e', "electron"]:
            channel_names = self.current_df_e.columns
            SOLO_EPT_CHANNELS_AMOUNT = 34
            SOLO_HET_CHANNELS_AMOUNT = 4
        if self.species in ['p', 'i', 'H', "proton", "ion"]:
            channel_names = self.current_df_i.columns
            SOLO_EPT_CHANNELS_AMOUNT = 64
            SOLO_HET_CHANNELS_AMOUNT = 36

        # Extract only the numbers from channel names
        if self.spacecraft == "solo":
            if self.sensor == "ept":
                channel_names = [name[1] for name in channel_names[:SOLO_EPT_CHANNELS_AMOUNT]]
                channel_numbers = np.array([int(name.split('_')[-1]) for name in channel_names])
            if self.sensor == "het":
                channel_names = [name[1] for name in channel_names[:SOLO_HET_CHANNELS_AMOUNT]]
                channel_numbers = np.array([int(name.split('_')[-1]) for name in channel_names])

        if self.spacecraft in ["sta", "stb"] or self.sensor == "erne":
            channel_numbers = np.array([int(name.split('_')[-1]) for name in channel_names])

        if self.sensor == "ephin":
            channel_numbers = np.array([int(name.split('E')[-1]) for name in channel_names])

        if self.sensor in ["ephin-5", "ephin-15"]:
            channel_numbers = [5, 15]

        if self.sensor == "isois-epihi":
            channel_numbers = np.array([int(name.split('_')[-1]) for name in channel_names])

        if self.sensor == "isois-epilo":
            channel_numbers = [int(name.split('_E')[-1].split('_')[0]) for name in channel_names]

        if self.sensor == "3dp":
            channel_numbers = [int(name.split('_')[1][-1]) for name in channel_names]

        # Remove any duplicates from the numbers array, since some dataframes come with, e.g., 'ch_2' and 'err_ch_2'
        channel_numbers = np.unique(channel_numbers)
        energy_strs = self.get_channel_energy_values("str")

        # SOHO/EPHIN returns one too many energy strs, because one of them is 'deactivated bc. or  failure mode D'
        if self.sensor == "ephin":
            energy_strs = energy_strs[:-1]

        # Assemble a pandas dataframe here for nicer presentation
        column_names = ("Channel", "Energy range")
        column_data = {
            column_names[0]: channel_numbers,
            column_names[1]: energy_strs}

        df = pd.DataFrame(data=column_data)

        # Set the channel number as the index of the dataframe
        df = df.set_index(column_names[0])

        # Finally display the dataframe such that ALL rows are shown
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               ):
            display(df)

    def save_and_update_rcparams(self, plotting_function: str):
        """
        A class method that saves the matplotlib rcParams that are preset before running a plotting routine, and then
        updates the rcParams to fit the plotting routine that is being run.

        Parameters:
        -----------
        plotting_function : str
                            The name of the plotting routine that is run, e.g., 'onset_tool', 'dynamic_spectrum' or 'tsa'
        """

        # The original rcParams set by the user prior to running function
        original_rcparams = rcParams.copy()

        # Here are listed all the possible dictionaries for the different plotting functions
        onset_options = {
            "axes.linewidth": 1.5,
            "font.size": 16
        }

        dyn_spec_options = {
            "axes.linewidth": 2.8,
            "font.size": 28 if self.radio_spacecraft is None else 20,
            "axes.titlesize": 32,
            "axes.labelsize": 28 if self.radio_spacecraft is None else 26,
            "xtick.labelsize": 28 if self.radio_spacecraft is None else 26,
            "ytick.labelsize": 20 if self.radio_spacecraft is None else 18,
            "pcolor.shading": "auto"
        }

        tsa_options = {
            "axes.linewidth": 1.5,
            "font.size": 12}

        options_dict = {
            "onset_tool": onset_options,
            "dynamic_spectrum": dyn_spec_options,
            "tsa": tsa_options}

        # Finally we update rcParams with the chosen plotting options
        rcParams.update(options_dict[plotting_function])

        return original_rcparams


def flux2series(flux, dates, cadence=None):
    """
    Converts an array of observed particle flux + timestamps into a pandas series
    with the desired cadence.

    Parameters:
    -----------
    flux: an array of observed particle fluxes
    dates: an array of corresponding dates/times
    cadence: str - desired spacing between the series elements e.g. '1s' or '5min'

    Returns:
    ----------
    flux_series: Pandas Series object indexed by the resampled cadence
    """

    # from pandas.tseries.frequencies import to_offset

    # set up the series object
    flux_series = pd.Series(flux, index=dates)

    # if no cadence given, then just return the series with the original
    # time resolution
    if cadence is not None:
        try:
            flux_series = flux_series.resample(cadence, origin='start').mean()
            flux_series.index = flux_series.index + pd.tseries.frequencies.to_offset(pd.Timedelta(cadence)/2)
        except ValueError:
            raise Warning(f"Your 'resample' option of [{cadence}] doesn't seem to be a proper Pandas frequency!")

    return flux_series


def bepicolombo_sixs_stack(path, date, side):
    # side is the index of the file here
    try:
        try:
            filename = f"{path}/sixs_phys_data_{date}_side{side}.csv"
            df = pd.read_csv(filename)
        except FileNotFoundError:
            # try alternative file name format
            filename = f"{path}/{date.strftime('%Y%m%d')}_side{side}.csv"
            df = pd.read_csv(filename)
            times = pd.to_datetime(df['TimeUTC'])
        # list comprehension because the method can't be applied onto the array "times"
        times = [t.tz_convert(None) for t in times]
        df.index = np.array(times)
        df = df.drop(columns=['TimeUTC'])
    except FileNotFoundError:
        print(f'Unable to open {filename}')
        df = pd.DataFrame()
        filename = ''
    return df, filename


def bepi_sixs_load(startdate, enddate, side, path):
    dates = pd.date_range(startdate, enddate)

    # read files into Pandas dataframes:
    df, file = bepicolombo_sixs_stack(path, startdate, side=side)
    if len(dates) > 1:
        for date in dates[1:]:
            t_df, file = bepicolombo_sixs_stack(path, date.date(), side=side)
            df = pd.concat([df, t_df])

    channels_dict = {"Energy_Bin_str": {'E1': '71 keV', 'E2': '106 keV', 'E3': '169 keV', 'E4': '280 keV', 'E5': '960 keV', 'E6': '2240 keV', 'E7': '8170 keV',
                                        'P1': '1.1 MeV', 'P2': '1.2 MeV', 'P3': '1.5 MeV', 'P4': '2.3 MeV', 'P5': '4.0 MeV', 'P6': '8.0 MeV', 'P7': '15.0 MeV', 'P8': '25.1 MeV', 'P9': '37.3 MeV'},
                     "Electron_Bins_Low_Energy": np.array([55, 78, 134, 235, 1000, 1432, 4904]),
                     "Electron_Bins_High_Energy": np.array([92, 143, 214, 331, 1193, 3165, 10000]),
                     "Ion_Bins_Low_Energy": np.array([0.001, 1.088, 1.407, 2.139, 3.647, 7.533, 13.211, 22.606, 29.246]),
                     "Ion_Bins_High_Energy": np.array([1.254, 1.311, 1.608, 2.388, 4.241, 8.534, 15.515, 28.413, 40.0])}
    return df, channels_dict


def calc_av_en_flux_sixs(df, channel, species):
    """
    This function averages the flux of two energy channels of BepiColombo/SIXS into a combined energy channel
    channel numbers counted from 1

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing HET data
    channel : int or list
        energy channel or list with first and last channel to be used
    species : string
        'e', 'electrons', 'p', 'protons'

    Returns
    -------
    flux: pd.DataFrame
        channel-averaged flux
    en_channel_string: str
        string containing the energy information of combined channel
    """

    # define constant geometric factors
    GEOMFACTOR_PROT8 = 5.97E-01
    GEOMFACTOR_PROT9 = 4.09E+00
    GEOMFACTOR_ELEC5 = 1.99E-02
    GEOMFACTOR_ELEC6 = 1.33E-01
    GEOMFACTOR_PROT_COMB89 = 3.34
    GEOMFACTOR_ELEC_COMB56 = 0.0972

    if species in ['p', 'protons']:
        if channel == [8, 9]:
            countrate = df['P8'] * GEOMFACTOR_PROT8 + df['P9'] * GEOMFACTOR_PROT9
            flux = countrate / GEOMFACTOR_PROT_COMB89
            en_channel_string = '37 MeV'
        else:
            print('No valid channel combination selected.')
            flux = pd.Series()
            en_channel_string = ''

    if species in ['e', 'electrons']:
        if channel == [5, 6]:
            countrate = df['E5'] * GEOMFACTOR_ELEC5 + df['E6'] * GEOMFACTOR_ELEC6
            flux = countrate / GEOMFACTOR_ELEC_COMB56
            en_channel_string = '1.4 MeV'
        else:
            print('No valid channel combination selected.')
            flux = pd.Series()
            en_channel_string = ''

    return flux, en_channel_string


"""
inf_inj_time.py
"""
SOLAR_ROT = sconst.get('sidereal rotation rate').to(u.rad/u.s)


def get_sun_coords(time='now'):
    '''
    Gets the astropy Sun coordinates.

    Args:
        time (datetime.datetime): time at which coordinates are fetched.

    Returns:
        sun coordinates.
    '''

    return get_horizons_coord("Sun", time=time)


def radial_distance_to_sun(spacecraft, time='now'):
    '''
    Gets the 3D radial distance of a spacecraft to the Sun.
    3D here means that it's the real spatial distance and not
    a projection on, say, the solar equatorial plane.

    Args:
        spacecraft (str): spacecraft to look for.
        time (datetime.datetime): time at which to look for.

    Returns:
        astropy units: radial distance.
    '''

    sc_coords = get_horizons_coord(spacecraft, time)

    return sc_coords.separation_3d(get_sun_coords(time=time))


def calc_spiral_length(radial_dist, sw_speed):
    '''
    Calculates the Parker spiral length from the Sun up to a given radial distance.

    Args:
        radial_dist (astropy units): radial distance to the Sun.
        sw_speed (astropy units): solar wind speed.

    Returns:
        astropy units: Parker spiral length.
    '''

    temp_const = ((SOLAR_ROT/sw_speed)*(radial_dist.to(u.km)-const.R_sun)).value
    sqrt_temp_const = sqrt(temp_const**2 + 1)

    return 0.5*u.rad * (sw_speed/SOLAR_ROT) * (temp_const*sqrt_temp_const + log(temp_const + sqrt_temp_const))


def calc_particle_speed(mass, kinetic_energy):
    '''
    Calculates the relativistic particle speed.

    Args:
        mass (astropy units): mass of the particle.
        kinetic_energy (astropy units): kinetic energy of the particle.

    Returns:
        astropy units: relativistic particle speed.
    '''

    gamma = sqrt(1 - (mass*const.c**2/(kinetic_energy + mass*const.c**2))**2)

    return gamma*const.c


def inf_inj_time(spacecraft, onset_time, species, kinetic_energy, sw_speed):
    '''
    Calculates the inferred injection time of a particle (electron or proton) from the Sun,
    given a detection time at some spacecraft.

    Args:
        spacecraft (str): name of the spacecraft.
        onset_time (datetime.datetime): time of onset/detection.
        species (str): particle species, 'p' or 'e'.
        kinetic_energy (astropy units): kinetic energy of particle. If no unit is supplied, is converted to MeV.
        sw_speed (astropy units): solar wind speed. If no unit is supplied, is converted to km/s.

    Returns:
        datetime.datetime: inferred injection time.
    '''

    if not type(kinetic_energy)==u.quantity.Quantity:
        kinetic_energy = kinetic_energy * u.MeV

    if not type(sw_speed)==u.quantity.Quantity:
        sw_speed = sw_speed * u.km/u.s

    mass_dict = {'p': const.m_p,
                 'e': const.m_e
                 }

    radial_distance = radial_distance_to_sun(spacecraft, time=onset_time)

    spiral_length = calc_spiral_length(radial_distance, sw_speed)
    particle_speed = calc_particle_speed(mass_dict[species], kinetic_energy)

    travel_time = spiral_length/particle_speed
    travel_time = travel_time.to(u.s)

    return onset_time - datetime.timedelta(seconds=travel_time.value), spiral_length.to(u.AU)
