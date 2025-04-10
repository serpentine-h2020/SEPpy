import copy
import os
import datetime
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as const
from sunpy.coordinates import get_horizons_coord
from matplotlib import rcParams
from matplotlib.dates import DateFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib.offsetbox import AnchoredText
from seppy.loader.psp import calc_av_en_flux_PSP_EPIHI, calc_av_en_flux_PSP_EPILO, psp_isois_load
from seppy.loader.soho import calc_av_en_flux_ERNE, soho_load
from seppy.loader.solo import epd_load
from seppy.loader.stereo import calc_av_en_flux_HET as calc_av_en_flux_ST_HET
from seppy.loader.stereo import calc_av_en_flux_SEPT, stereo_load
from seppy.loader.wind import wind3dp_load
from seppy.util import bepi_sixs_load, calc_av_en_flux_sixs, custom_warning, flux2series, resample_df


# This is to get rid of this specific warning:
# /home/user/xyz/serpentine/notebooks/sep_analysis_tools/read_swaves.py:96: UserWarning: The input coordinates to pcolormesh are interpreted as
# cell centers, but are not monotonically increasing or decreasing. This may lead to incorrectly calculated cell edges, in which
# case, please supply explicit cell edges to pcolormesh.
# colormesh = ax.pcolormesh( time_arr, freq[::-1], data_arr[::-1], vmin = 0, vmax = 0.5*np.max(data_arr), cmap = 'inferno' )
warnings.filterwarnings(action="ignore",
                        message="The input coordinates to pcolormesh are interpreted as cell centers, but are not monotonically increasing or \
                        decreasing. This may lead to incorrectly calculated cell edges, in which case, please supply explicit cell edges to pcolormesh.",
                        category=UserWarning)

STEREO_SEPT_VIEWINGS = ("sun", "asun", "north", "south")
WIND_3DP_VIEWINGS = ("omnidirectional", '0', '1', '2', '3', '4', '5', '6', '7')
SOLO_EPT_VIEWINGS = ("sun", "asun", "north", "south")
SOLO_HET_VIEWINGS = ("sun", "asun", "north", "south")
SOLO_STEP_VIEWINGS = ("Pixel averaged", "Pixel 1", "Pixel 2", "Pixel 3", "Pixel 4", "Pixel 5", "Pixel 6",
                      "Pixel 7", "Pixel 8", "Pixel 9", "Pixel 10", "Pixel 11", "Pixel 12", "Pixel 13",
                      "Pixel 14", "Pixel 15")
PSP_EPILO_VIEWINGS = ('3', '7')
PSP_EPIHI_VIEWINGS = ('A', 'B')


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
        if species in ("electrons", "electron"):
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

        # Sets the self.viewing to the given viewing
        self.update_viewing(viewing=viewing)

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

        # # Check that the data that was loaded is valid. If not, give a warning.
        self.validate_data()

        # Download radio cdf files ONLY if asked to
        if self.radio_spacecraft is not None:
            from seppy.tools.swaves import get_swaves
            self.radio_files = get_swaves(start_date, end_date)

    def validate_data(self):
        """
        Provide an error msg if this object is initialized with a combination that yields invalid data products.
        """

        # SolO/STEP data before 22 Oct 2021 is not supported yet for non-'Pixel averaged' viewing
        warn_mess_step_pixels_old = "SolO/STEP data is not included yet for individual Pixels for dates preceding Oct 22, 2021. Only 'Pixel averaged' is supported."
        if self.spacecraft == "solo" and self.sensor == "step":
            if self.start_date < pd.to_datetime("2021-10-22").date():
                if not self.viewing == 'Pixel averaged':
                    # when 'viewing' is undefined, only give a warning; if it's wrong defined, abort with warning
                    if not self.viewing:
                        # warnings.warn(message=warn_mess_step_pixels_old)
                        custom_warning(message=warn_mess_step_pixels_old)
                    else:
                        raise Warning(warn_mess_step_pixels_old)

        # Electron data for SolO/STEP is removed for now (Feb 2024, JG)
        if self.spacecraft == "solo" and self.sensor == "step" and self.species.lower()[0] == 'e':
            raise Warning("SolO/STEP electron data is not implemented yet!")

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

        invalid_viewing_msg = f"{viewing} is an invalid viewing direction for {self.spacecraft}/{self.sensor}!"

        if self.spacecraft != "wind":

            # Validate viewing here. It may be nonsensical and that affects choose_data() and print_energies().
            if self.spacecraft in ("sta", "stb"):
                if self.sensor == "sept" and viewing not in STEREO_SEPT_VIEWINGS:
                    raise ValueError(invalid_viewing_msg)
                if self.sensor == "het" and viewing is not None:
                    raise ValueError(invalid_viewing_msg)

            if self.spacecraft == "solo":
                if self.sensor == "step" and viewing not in SOLO_STEP_VIEWINGS:
                    raise ValueError(invalid_viewing_msg)
                if self.sensor == "ept" and viewing not in SOLO_EPT_VIEWINGS:
                    raise ValueError(invalid_viewing_msg)
                if self.sensor == "het" and viewing not in SOLO_HET_VIEWINGS:
                    raise ValueError(invalid_viewing_msg)

            if self.spacecraft == "psp":
                if self.sensor == "isois-epilo" and viewing not in PSP_EPILO_VIEWINGS:
                    raise ValueError(invalid_viewing_msg)
                if self.sensor == "isois-epihi" and viewing not in PSP_EPIHI_VIEWINGS:
                    raise ValueError(invalid_viewing_msg)

            if self.spacecraft == "soho":
                if viewing is not None:
                    raise ValueError(invalid_viewing_msg)

            # Finally set validated viewing
            self.viewing = viewing

        else:
            # Wind/3DP viewing directions are omnidirectional, section 0, section 1... section 7.
            # This catches the number or the word if omnidirectional
            try:
                sector_direction = viewing.split(" ")[-1]
            # AttributeError is caused by calling None.split()
            except AttributeError:
                raise ValueError(invalid_viewing_msg)

            if sector_direction not in WIND_3DP_VIEWINGS:
                raise ValueError(invalid_viewing_msg)

            self.viewing = sector_direction

    # I suggest we at some point erase the arguments ´spacecraft´ and ´threshold´ due to them not being used.
    # `viewing` and `autodownload` are actually the only necessary input variables for this function, the rest
    # are class attributes, and should probably be cleaned up at some point
    def load_data(self, spacecraft, sensor, viewing, data_level,
                  autodownload=True, threshold=None):

        if self.spacecraft == 'solo':

            if self.sensor in ("ept", "het"):
                df_i, df_e, meta = epd_load(sensor=sensor,
                                            viewing=viewing,
                                            level=data_level,
                                            startdate=self.start_date,
                                            enddate=self.end_date,
                                            path=self.data_path,
                                            autodownload=autodownload)

                return df_i, df_e, meta

            elif self.sensor == "step":
                df, meta = epd_load(sensor=sensor,
                                    viewing="None",
                                    level=data_level,
                                    startdate=self.start_date,
                                    enddate=self.end_date,
                                    path=self.data_path,
                                    autodownload=autodownload)

                return df, meta

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
                                                           pos_timestamp="center",
                                                           path=self.data_path)
                    df_e, channels_dict_df_e = [], []

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
                                                           pos_timestamp="center",
                                                           path=self.data_path)

                    df_i, channels_dict_df_i = [], []

                    return df_i, df_e, channels_dict_df_i, channels_dict_df_e

            if self.sensor == 'het':
                df, meta = stereo_load(instrument=self.sensor,
                                       startdate=self.start_date,
                                       enddate=self.end_date,
                                       spacecraft=self.spacecraft,
                                       resample=None,
                                       pos_timestamp="center",
                                       path=self.data_path)

                return df, meta

        if self.spacecraft.lower() == 'soho':
            if self.sensor == 'erne':
                df, meta = soho_load(dataset="SOHO_ERNE-HED_L2-1MIN",
                                     startdate=self.start_date,
                                     enddate=self.end_date,
                                     path=self.data_path,
                                     resample=None,
                                     pos_timestamp="center")

                return df, meta

            if self.sensor == 'ephin':
                df, meta = soho_load(dataset="SOHO_COSTEP-EPHIN_L2-1MIN",
                                     startdate=self.start_date,
                                     enddate=self.end_date,
                                     path=self.data_path,
                                     resample=None,
                                     pos_timestamp="center")

                return df, meta

            if self.sensor in ("ephin-5", "ephin-15"):

                dataset = "ephin_flux_2020-2022.csv"

                if os.path.isfile(f"{self.data_path}{dataset}"):
                    df = pd.read_csv(f"{self.data_path}{dataset}", index_col="date", parse_dates=True)
                    # set electron flux to nan if the ratio of proton-proxy counts to correspondning electron counts is >0.1
                    df['E5'][df['p_proxy (1.80-2.00 MeV)']/df['0.45-0.50 MeV (E5)'] >= 0.1] = np.nan
                    df['E15'][df['p_proxy (1.80-2.00 MeV)']/df['0.70-1.10 MeV (E15)'] >= 0.1] = np.nan
                else:
                    raise Warning(f"File {dataset} not found at {self.data_path}! Please verify that 'data_path' is correct.")
                meta = {"E5": "0.45 - 0.50 MeV",
                        "E15": "0.70 - 1.10 MeV"}

                # TODO:
                # - add resample_df here?
                # - add pos_timestamp here

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

                return df_omni_i, df_omni_e, df_i, df_e, meta_i, meta_e

        if self.spacecraft.lower() == 'psp':
            if self.sensor.lower() == 'isois-epihi':
                df, meta = psp_isois_load(dataset='PSP_ISOIS-EPIHI_L2-HET-RATES60',
                                          startdate=self.start_date,
                                          enddate=self.end_date,
                                          path=self.data_path,
                                          resample=None)

                return df, meta
            if self.sensor.lower() == 'isois-epilo':
                df, meta = psp_isois_load(dataset='PSP_ISOIS-EPILO_L2-PE',
                                          startdate=self.start_date,
                                          enddate=self.end_date,
                                          path=self.data_path,
                                          resample=None,
                                          epilo_channel='F',
                                          epilo_threshold=self.threshold)

                return df, meta

        if self.spacecraft.lower() == 'bepi':
            df, meta = bepi_sixs_load(startdate=self.start_date,
                                      enddate=self.end_date,
                                      side=viewing,
                                      path=self.data_path,
                                      pos_timestamp='center')
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
                    self.load_data(self.spacecraft, self.sensor, self.viewing,
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
            if not viewing:
                raise Exception("For this operation, the instrument's 'viewing' direction must be defined in the call of 'Event'!")

            elif viewing == 'sun':

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

            elif "Pixel" in viewing:

                # All viewings are contained in the same dataframe, choose the pixel (viewing) here
                pixel = self.viewing.split(' ')[1]

                # Pixel info is in format NN in the dataframe, 1 -> 01 while 12 -> 12
                if len(pixel) == 1:
                    pixel = f"0{pixel}"

                # Pixel length more than 2 means "averaged" -> called "Avg" in the dataframe
                elif len(pixel) > 2:
                    pixel = "Avg"

                self.current_df_i = self.df_step[[col for col in self.df_step.columns if f"Magnet_{pixel}_Flux" in col]]
                self.current_df_e = self.df_step[[col for col in self.df_step.columns if f"Electron_{pixel}_Flux" in col]]
                self.current_energies = self.energies_step

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
                en_channel_string = en_str[en_channel[0]].flat[0].split()[0] + ' - '\
                    + en_str[en_channel[-1]].flat[0].split()[2] + ' ' +\
                    en_str[en_channel[-1]].flat[0].split()[3]

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
                en_channel_string = en_str[en_channel[0]].flat[0].split()[0] + ' - '\
                    + en_str[en_channel[-1]].flat[0].split()[2] + ' '\
                    + en_str[en_channel[-1]].flat[0].split()[3]

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
            norm_channel[i] = (flux_series.iloc[i]-ma)/sigma

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

            ylim = [np.nanmin(flux_series[flux_series > 0]/2),
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
                   label="Background", alpha=0.5)

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
                                      f"Peak flux: {df_flux_peak['flux'].iloc[0]:.2E}",
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
                        Pandas-compatible time string to average data. e.g. '10s' for 10 seconds or '2min' for 2 minutes.
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
                background_warning = "Your background_range is separated from plot_range by over a day. If this was intentional you may ignore this warning."
                # warnings.warn(message=background_warning)
                custom_warning(message=background_warning)

        if (self.spacecraft[:2].lower() == 'st' and self.sensor == 'sept') \
                or (self.spacecraft.lower() == 'psp' and self.sensor.startswith('isois')) \
                or (self.spacecraft.lower() == 'solo' and self.sensor == 'step') \
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

        # Check that the data that was loaded is valid. If not, abort with warning.
        self.validate_data()

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

            elif self.sensor == "step":

                if len(channels) > 1:
                    not_implemented_msg = "Multiple channel averaging not yet supported for STEP! Please choose only one channel."
                    raise Exception(not_implemented_msg)

                en_channel_string = self.get_channel_energy_values("str")[channels[0]]

                if self.species in ('p', 'i'):
                    channel_id = self.current_df_i.columns[channels[0]]
                    df_flux = pd.DataFrame(data={
                        "flux": self.current_df_i[channel_id]
                    }, index=self.current_df_i.index)

                elif self.species == 'e':
                    channel_id = self.current_df_e.columns[channels[0]]
                    df_flux = pd.DataFrame(data={
                        "flux": self.current_df_e[channel_id]
                    }, index=self.current_df_e.index)

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

            df_averaged = resample_df(df=df_flux, resample=resample_period)

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

    def dynamic_spectrum(self, view, cmap: str = 'magma', xlim: tuple = None, resample: str = None, save: bool = False,
                         other=None) -> None:
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

        def get_yaxis_bin_boundaries(e_lows, e_highs, y_multiplier, is_solohetions):
            """
            Helper function to produce the bin boundaries for dynamic spectrum y-axis.
            """

            # For any other sc+instrument combination than solo+HET in the current version, there is no need to complicate setting bin boundaries
            if not is_solohetions:
                return np.append(e_lows, e_highs[-1]) * y_multiplier

            # Init the boundaries. For SolO/HET there are more boundaries than channels
            yaxis_bin_boundaries = np.zeros(len(e_lows)+2)
            yaxis_idx = 0
            chooser_idx = yaxis_idx
            while yaxis_idx < len(yaxis_bin_boundaries)-1:  # this loop will not go to the final index of the array

                # Set the first boundary as simply the first val of lower energy boundaries and continue
                if yaxis_idx==0:
                    yaxis_bin_boundaries[yaxis_idx] = e_lows[chooser_idx]
                    yaxis_idx += 1
                    chooser_idx += 1
                    continue

                # If the lower boundary now is the same as the last bins higher boundary, then set that as the boundary
                if e_lows[chooser_idx] == e_highs[chooser_idx-1]:
                    yaxis_bin_boundaries[yaxis_idx] = e_lows[chooser_idx]
                    yaxis_idx += 1
                    chooser_idx += 1

                # ...if not, set the last higher boundary as the boundary, the next lower boundary as the next boundary and continue business as usual
                else:
                    yaxis_bin_boundaries[yaxis_idx] = e_highs[chooser_idx-1]
                    yaxis_bin_boundaries[yaxis_idx+1] = e_lows[chooser_idx]

                    yaxis_idx += 2
                    chooser_idx += 1

            # Finally the last boundary is the final boundary of e_highs:
            yaxis_bin_boundaries[-1] = e_highs[-1]

            return yaxis_bin_boundaries * y_multiplier

        def combine_grids_and_ybins(grid, grid1, y_arr, y_arr1):
            # TODO: Which bin exactly is removed here? HET? EPT? (JG)

            # solo/het lowest electron channel partially overlaps with ept highest channel -> erase the "extra" bin where overlapping hapens
            if self.spacecraft == "solo" and (self.sensor == "het" or other.sensor == "het") and self.species in ("electrons", "electron", 'e'):

                grid1 = np.append(grid, grid1, axis=0)[:-1]

                # This deletes the first entry of y_arr1
                y_arr1 = np.delete(y_arr1, 0, axis=0)
                y_arr1 = np.append(y_arr, y_arr1)

            # There is a gap between the highest solo/ept proton channel and the lowest het ion channel -> add extra row to grid
            # filled with nans to compensate
            elif self.spacecraft == "solo" and (self.sensor == "het" or other.sensor == "het") and self.species == 'p':

                nans = np.array([[np.nan for i in range(len(grid[0]))]])
                grid = np.append(grid, nans, axis=0)
                grid1 = np.append(grid, grid1, axis=0)[:-2]
                y_arr1 = np.append(y_arr, y_arr1)

            else:

                grid1 = np.append(grid, grid1, axis=0)
                y_arr1 = np.append(y_arr, y_arr1)

            return grid1, y_arr1

        # Event attributes
        spacecraft = self.spacecraft.lower()
        instrument = self.sensor.lower()
        species = self.species

        # Boolean value for checking if y-axis requires a white stripe
        is_solohetions = (spacecraft == "solo" and instrument == "het" and species == 'p')

        # This method has to be run before doing anything else to make sure that the viewing is correct
        self.choose_data(view)

        # Check that the data that was loaded is valid. If not, abort with warning.
        self.validate_data()

        if self.spacecraft == "solo":

            if instrument == "step":
                # custom_warning('The lower STEP energy channels are partly overlapping, which is not correctly implemented at the moment!')
                raise Warning('SolO/STEP is not implemented yet in the dynamic spectrum tool!')

                # All viewings are contained in the same dataframe, choose the pixel (viewing) here
                pixel = self.viewing.split(' ')[1]

                # Pixel info is in format NN in the dataframe, 1 -> 01 while 12 -> 12
                if len(pixel) == 1:
                    pixel = f"0{pixel}"

                # Pixel length more than 2 means "averaged" -> called "Avg" in the dataframe
                elif len(pixel) > 2:
                    pixel = "Avg"

                if species in ("electron", 'e'):
                    particle_data = self.df_step[[col for col in self.df_step.columns if f"Electron_{pixel}_Flux" in col]]
                    s_identifier = "electrons"

                # Step's "Magnet" channel deflects electrons -> measures all positive ions
                else:
                    particle_data = self.df_step[[col for col in self.df_step.columns if f"Magnet_{pixel}_Flux" in col]]
                    s_identifier = "ions"

            # EPT and HET data come in almost identical containers, they need not be differentiated
            elif species in ("electron", 'e'):
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
                # Here drop the E300 channel altogether from the dataframe if the data is produced after Oct 4, 2017,
                # for it contains no valid data. Keyword axis==1 refers to the columns axis.
                if self.start_date > pd.to_datetime("2017-10-04").date():
                    particle_data = particle_data.drop("E300", axis=1)

                s_identifier = "electrons"
                # raise Warning('SOHO/EPHIN is not implemented yet in the dynamic spectrum tool!')

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
        LOW_ENERGY_SENSORS = ("sept", "step", "ept")

        # For a single instrument, check if low or high energy instrument, but for joint dynamic spectrum
        # always use MeVs as the unit, because the y-axis is going to range over a large number of values
        if not other:
            if instrument in LOW_ENERGY_SENSORS:
                y_multiplier = 1e-3  # keV
                y_unit = "keV"
            else:
                y_multiplier = 1e-6  # MeV
                y_unit = "MeV"
        else:
            y_multiplier = 1e-6  # MeV
            y_unit = "MeV"

        # Resample only if requested
        if resample is not None:
            particle_data = resample_df(df=particle_data, resample=resample)

        if xlim is None:
            df = particle_data[:]
            t_start, t_end = df.index[0], df.index[-1]
        else:
            # td is added to the start and the end to avert white pixels at the end of the plot
            td_str = resample if resample is not None else '0s'
            td = pd.Timedelta(value=td_str)
            t_start, t_end = pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1])
            df = particle_data.loc[(particle_data.index >= (t_start-td)) & (particle_data.index <= (t_end+td))]

        # In practice this seeks the date on which the highest flux is observed
        date_of_event = df.iloc[np.argmax(df[df.columns[0]])].name.date()

        # Assert time and channel bins
        time = df.index

        # The low and high ends of each energy channel
        e_lows, e_highs = self.get_channel_energy_values()  # this function return energy in eVs

        # The mean energy of each channel in eVs
        mean_energies = np.sqrt(np.multiply(e_lows, e_highs))

        # Boundaries of plotted bins in keVs are the y-axis:
        y_arr = get_yaxis_bin_boundaries(e_lows, e_highs, y_multiplier, is_solohetions)

        # Set image pixel length and height
        image_len = len(time)
        image_hei = len(y_arr)-1

        # Init the grid
        grid = np.zeros((image_len, image_hei))

        # Display energy in MeVs -> multiplier squared is 1e-6*1e-6 = 1e-12
        ENERGY_MULTIPLIER_SQUARED = 1e-12

        # Assign grid bins -> intensity * energy^2
        if is_solohetions:
            for i, channel in enumerate(df):

                if i<5:
                    grid[:, i] = df[channel]*(mean_energies[i]*mean_energies[i]*ENERGY_MULTIPLIER_SQUARED)
                elif i==5:
                    grid[:, i] = np.nan
                    grid[:, i+1] = df[channel]*(mean_energies[i]*mean_energies[i]*ENERGY_MULTIPLIER_SQUARED)
                else:
                    grid[:, i+1] = df[channel]*(mean_energies[i]*mean_energies[i]*ENERGY_MULTIPLIER_SQUARED)

        else:
            for i, channel in enumerate(df):

                grid[:, i] = df[channel]*(mean_energies[i]*mean_energies[i]*ENERGY_MULTIPLIER_SQUARED)  # Intensity*Energy^2, and energy is in eV -> tranform to keV or MeV

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

        ax[DYN_SPEC_INDX].set_yscale('log')

        # Colorbar
        if not other:

            # Colormesh
            cplot = ax[DYN_SPEC_INDX].pcolormesh(time, y_arr, grid, shading='auto', cmap=cmap, norm=normscale)
            greymesh = ax[DYN_SPEC_INDX].pcolormesh(time, y_arr, maskedgrid, shading='auto', cmap='Greys', vmin=-1, vmax=1)

            cb = fig.colorbar(cplot, orientation='vertical', ax=ax[DYN_SPEC_INDX])
            clabel = r"Intensity $\cdot$ $E^{2}$" + "\n" + r"[MeV/(cm$^{2}$ sr s)]"
            cb.set_label(clabel)

            # y-axis settings
            ax[DYN_SPEC_INDX].set_ylim(np.nanmin(y_arr), np.nanmax(y_arr))
            ax[DYN_SPEC_INDX].set_yticks([yval for yval in y_arr])
            ax[DYN_SPEC_INDX].set_ylabel(f"Energy [{y_unit}]")

        # Format of y-axis: for single instrument use pretty numbers, for joint spectrum only powers of ten
        if not other:
            ax[DYN_SPEC_INDX].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        else:
            ax[DYN_SPEC_INDX].yaxis.set_major_formatter(ScalarFormatter(useMathText=False))

        # gets rid of minor ticks and labels
        ax[DYN_SPEC_INDX].yaxis.minorticks_off()
        ax[DYN_SPEC_INDX].yaxis.set_tick_params(length=12., width=2.0, which='major')

        # x-axis settings
        # ax[DYN_SPEC_INDX].set_xlabel("Time [HH:MM \nm-d]")
        ax[DYN_SPEC_INDX].xaxis_date()
        ax[DYN_SPEC_INDX].set_xlim(t_start, t_end)
        # ax[DYN_SPEC_INDX].xaxis.set_major_locator(mdates.HourLocator(interval = 1))
        # utc_dt_format1 = DateFormatter('%H:%M \n%m-%d')
        utc_dt_format1 = DateFormatter('%H:%M\n%b %d\n%Y')
        ax[DYN_SPEC_INDX].xaxis.set_major_formatter(utc_dt_format1)
        # ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 5))

        # Expand the spectrum to a second instrument (for now only for solo/ept + step or het)
        if other and self.sensor == "ept" and other.sensor == "het":

            is_solohetions = (other.sensor == "het" and species == 'p')

            # This method has to be run before doing anything else to make sure that the viewing is correct
            other.choose_data(other.viewing)

            # EPT and HET data come in almost identical containers, they need not be differentiated
            if species in ("electron", 'e'):
                particle_data1 = other.current_df_e["Electron_Flux"]
            else:
                try:
                    particle_data1 = other.current_df_i["Ion_Flux"]
                except KeyError:
                    particle_data1 = other.current_df_i["H_Flux"]

            # Resample only if requested
            if resample is not None:
                particle_data1 = resample_df(df=particle_data1, resample=resample)

            if xlim is None:
                df1 = particle_data1[:]
                # t_start, t_end = df.index[0], df.index[-1]
            else:
                # td is added to the start and the end to avert white pixels at the end of the plot
                td_str = resample if resample is not None else '0s'
                td = pd.Timedelta(value=td_str)
                t_start, t_end = pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1])
                df1 = particle_data1.loc[(particle_data1.index >= (t_start-td)) & (particle_data1.index <= (t_end+td))]

            # Assert time and channel bins
            time1 = df.index

            # The low and high ends of each energy channel
            e_lows1, e_highs1 = other.get_channel_energy_values()  # this function return energy in eVs

            # The mean energy of each channel in eVs
            mean_energies1 = np.sqrt(np.multiply(e_lows1, e_highs1))

            # Boundaries of plotted bins in keVs are the y-axis:
            y_arr1 = get_yaxis_bin_boundaries(e_lows1, e_highs1, y_multiplier, is_solohetions)

            # Set image pixel height (length was already set before)
            # For solohet+ions we do not subtract 1 here, because there is an energy gap between EPT highest channel and
            # HET lowest channel, hence requiring one "empty" bin in between
            image_hei1 = len(y_arr1)+1 if is_solohetions else len(y_arr1)

            # Init the grid
            grid1 = np.zeros((image_len, image_hei1))

            # Assign grid bins -> intensity * energy^2
            if is_solohetions:
                for i, channel in enumerate(df1):

                    if i<5:
                        grid1[:, i] = df1[channel]*(mean_energies1[i]*mean_energies1[i]*ENERGY_MULTIPLIER_SQUARED)
                    elif i==5:
                        grid1[:, i] = np.nan
                        grid1[:, i+1] = df1[channel]*(mean_energies1[i]*mean_energies1[i]*ENERGY_MULTIPLIER_SQUARED)
                    else:
                        grid1[:, i+1] = df1[channel]*(mean_energies1[i]*mean_energies1[i]*ENERGY_MULTIPLIER_SQUARED)

            else:
                for i, channel in enumerate(df1):

                    grid1[:, i] = df1[channel]*(mean_energies1[i]*mean_energies1[i]*ENERGY_MULTIPLIER_SQUARED)  # Intensity*Energy^2, and energy is in eV -> transform to keV or MeV

            # Finally cut the last entry and transpose the grid1 so that it can be plotted correctly
            grid1 = grid1[:-1, :]
            grid1 = grid1.T

            # grids and y-axis has to be fused together so they can be plotted in the same colormesh
            grid1, y_arr1 = combine_grids_and_ybins(grid, grid1, y_arr, y_arr1)

            maskedgrid1 = np.where(grid1 == 0, 0, 1)
            maskedgrid1= np.ma.masked_where(maskedgrid1 == 1, maskedgrid1)

            # return time1, y_arr1, grid1
            # Colormesh
            cplot1 = ax[DYN_SPEC_INDX].pcolormesh(time1, y_arr1, grid1, shading='auto', cmap=cmap, norm=normscale)
            greymesh1 = ax[DYN_SPEC_INDX].pcolormesh(time1, y_arr1, maskedgrid1, shading='auto', cmap='Greys', vmin=-1, vmax=1)

            # Updating the colorbar
            cb = fig.colorbar(cplot1, orientation='vertical', ax=ax[DYN_SPEC_INDX])
            clabel = r"Intensity $\cdot$ $E^{2}$" + "\n" + r"[MeV/(cm$^{2}$ sr s)]"
            cb.set_label(clabel)

            # y-axis settings
            ax[DYN_SPEC_INDX].set_yscale('log')
            ax[DYN_SPEC_INDX].set_ylim(np.nanmin(y_arr), np.nanmax(y_arr1))

            # Set a rougher tickscale
            energy_tick_powers = (-1, 1, 3) if species in ("electron", 'e') else (-1, 2, 4)
            yticks = np.logspace(start=energy_tick_powers[0], stop=energy_tick_powers[1], num=energy_tick_powers[2])

            # First one sets the ticks in place and the second one enlarges the tick labels (not the ticks, the numbers next to them)
            ax[DYN_SPEC_INDX].set_yticks([yval for yval in yticks])
            ax[DYN_SPEC_INDX].tick_params(axis='y', labelsize=32)

            ax[DYN_SPEC_INDX].set_ylabel(f"Energy [{y_unit}]")

            # Introduce minor ticks back
            ax[DYN_SPEC_INDX].yaxis.set_tick_params(length=8., width=1.2, which='minor')

            fig.set_size_inches((27, 18))

        # Title
        if view is not None and other:
            title = f"{spacecraft.upper()}/{instrument.upper()}+{other.sensor.upper()} ({view}) {s_identifier}, {date_of_event}"
        elif view:
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
        from IPython.display import display

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

        # Check that the data that was loaded is valid. If not, abort with warning.
        self.validate_data()

        if self.spacecraft == "solo":
            if self.sensor == "step":

                if species in ("electron", 'e'):
                    particle_data = self.current_df_e
                    s_identifier = "electrons"
                else:
                    particle_data = self.current_df_i
                    s_identifier = "ions"

            else:

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

            sc_identifier = "Solar Orbiter"

        if self.spacecraft[:2] == "st":
            if species in ("electron", 'e'):
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
        min_slider_val, max_slider_val = 0.0, 10

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
            text.set_text(f"R={radial_distance_value:.2f} AU\nL = {np.round(slider.value, 2)} AU")

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

            # STEP, ETP and HET energies are in the same object
            energy_dict = self.current_energies

            if self.species in ("electron", 'e'):
                energy_ranges = energy_dict["Electron_Bins_Text"]

            else:
                p_identifier =  "Ion_Bins_Text" if self.sensor == "ept" else "H_Bins_Text" if self.sensor == "het" else "Bins_Text"
                energy_ranges = energy_dict[p_identifier]

            # Each element in the list is also a list with len==1 for cdflib < 1.3.3, so fix that
            energy_ranges = energy_ranges.flatten()

        if self.spacecraft[:2] == "st":

            # STEREO/SEPT energies come in two different objects
            if self.sensor == "sept":
                if self.species in ("electron", 'e'):
                    energy_df = self.current_e_energies
                else:
                    energy_df = self.current_i_energies

                energy_ranges = energy_df["ch_strings"].values

            # STEREO/HET energies all in the same dictionary
            else:
                energy_dict = self.current_energies

                if self.species in ("electron", 'e'):
                    energy_ranges = energy_dict["Electron_Bins_Text"]
                else:
                    energy_ranges = energy_dict["Proton_Bins_Text"]

                # Each element in the list is also a list with len==1 for cdflib < 1.3.3, so fix that
                energy_ranges = energy_ranges.flatten()

        if self.spacecraft == "soho":
            if self.sensor.lower() == "erne":
                energy_ranges = self.current_energies["channels_dict_df_p"]["ch_strings"].values
            if self.sensor.lower() == "ephin":
                # Choose only the first 4 channels (E150, E300, E1300 and E3000)
                # These are the only electron channels (rest are p and He), and we
                # use only electron data here.
                energy_ranges = [val for val in self.current_energies.values()][:4]
            if self.sensor.lower() in ("ephin-5", "ephin-15"):
                energy_ranges = [value for _, value in self.current_energies.items()]

        if self.spacecraft == "psp":
            energy_dict = self.meta

            if self.sensor == "isois-epihi":
                if self.species == 'e':
                    energy_ranges = energy_dict["Electrons_ENERGY_LABL"]
                if self.species == 'p':
                    energy_ranges = energy_dict["H_ENERGY_LABL"]

                # Each element in the list is also a list with len==1 for cdflib < 1.3.3, so fix that
                energy_ranges = energy_ranges.flatten()

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
                    # will not work anymore since the number of channels and channel energy ranges won't be the same.
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

        from IPython.display import display

        # This has to be run first, otherwise self.current_df does not exist
        # Note that PSP will by default have its viewing=='all', which does not yield proper dataframes
        if self.viewing != 'all':
            if self.spacecraft == 'solo' and not self.viewing:
                raise Warning("For this operation the instrument's 'viewing' direction must be defined in the call of 'Event'! Please define and re-run.")
                return
            else:
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

            if self.sensor == "step":
                channel_names = list(channel_names)
                channel_numbers = np.array([int(name.split('_')[-1]) for name in channel_names])

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

        # The following behaviour has been fixed upstream. Keeping this here for
        # now in case someone is missing it.
        # # SOHO/EPHIN returns one too many energy strs, because one of them is 'deactivated bc. or  failure mode D'
        # # if self.sensor == "ephin":
        # #     energy_strs = energy_strs[:-1]

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
        return

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
