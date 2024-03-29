"""
A library to run the interactive user interface in SEP event onset determination notebooks.

@Author: Christian Palmroos <chospa@utu.fi>
"""

import ipywidgets as widgets

# a list of available spacecraft:
list_of_sc = ["PSP", "SOHO", "Solar Orbiter", "STEREO-A", "STEREO-B", "Wind"]

stereo_instr = ["HET", "SEPT"]  # ["LET", "SEPT", "HET"]
solo_instr = ["EPT", "HET", "STEP"]
bepi_instr = ["SIXS-P"]
soho_instr = ["EPHIN", "ERNE-HED"]
psp_instr = ["isois-epihi", "isois-epilo"]
wind_instr = ["3DP"]

sensor_dict = {
    "STEREO-A": stereo_instr,
    "STEREO-B": stereo_instr,
    "Solar Orbiter": solo_instr,
    "Bepicolombo": bepi_instr,
    "SOHO": soho_instr,
    "PSP": psp_instr,
    "Wind": wind_instr
}

view_dict = {
    ("STEREO-A", "SEPT"): ("sun", "asun", "north", "south"),
    ("STEREO-B", "SEPT"): ("sun", "asun", "north", "south"),
    ("Solar Orbiter", "STEP"): ("Pixel averaged", "Pixel 1", "Pixel 2", "Pixel 3", "Pixel 4", "Pixel 5", "Pixel 6", "Pixel 7", "Pixel 8", "Pixel 9", "Pixel 10",
                                "Pixel 11", "Pixel 12", "Pixel 13", "Pixel 14", "Pixel 15"),
    ("Solar Orbiter", "EPT"): ("sun", "asun", "north", "south"),
    ("Solar Orbiter", "HET"): ("sun", "asun", "north", "south"),
    ("Bepicolombo", "SIXS-P"): (0, 1, 2, 3, 4),
    ("PSP", "isois-epihi"): ("A", "B"),
    ("PSP", "isois-epilo"): ('3', '7'),  # ('0', '1', '2', '3', '4', '5', '6', '7')
    ("Wind", "3DP"): ('omnidirectional', 'sector 0', 'sector 1', 'sector 2', 'sector 3', 'sector 4', 'sector 5', 'sector 6', 'sector 7')
}

species_dict = {
    ("STEREO-A", "LET"): ("protons", "electrons"),
    ("STEREO-A", "SEPT"): ("ions", "electrons"),
    ("STEREO-A", "HET"): ("protons", "electrons"),
    ("STEREO-B", "LET"): ("protons", "electrons"),
    ("STEREO-B", "SEPT"): ("ions", "electrons"),
    ("STEREO-B", "HET"): ("protons", "electrons"),
    ("Solar Orbiter", "STEP"): ("ions",),  # , "electrons"),
    ("Solar Orbiter", "EPT"): ("ions", "electrons"),
    ("Solar Orbiter", "HET"): ("protons", "electrons"),
    ("Bepicolombo", "SIXS-P"): ("protons", "electrons"),
    ("SOHO", "ERNE-HED"): ("protons",),
    ("SOHO", "EPHIN"): ("electrons",),
    ("PSP", "isois-epihi"): ("protons", "electrons"),
    ("PSP", "isois-epilo"): ("electrons",),
    ("Wind", "3DP"): ("protons", "electrons")
}

radio_dict = {
    "None": None,
    "STEREO-A": ("ahead", "STEREO-A"),
    "STEREO-B": ("behind", "STEREO-B"),
    # "WIND (Coming soon!)": ("wind", "WIND")  # TODO: re-add when supported!
}

# Drop-downs for dynamic particle spectrum:
spacecraft_drop = widgets.Dropdown(options=list_of_sc,
                                   description="Spacecraft:",
                                   disabled=False,
                                   )

sensor_drop = widgets.Dropdown(options=sensor_dict[spacecraft_drop.value],
                               description="Sensor:",
                               disabled=False,
                               )

view_drop = widgets.Dropdown(options=view_dict[(spacecraft_drop.value, sensor_drop.value)],
                             description="Viewing:",
                             disabled=False
                             )

species_drop = widgets.Dropdown(options=species_dict[(spacecraft_drop.value, sensor_drop.value)],
                                description="Species:",
                                disabled=False,
                                )


# A button to enable radio spectrum (Leave this out for now, sincde it doesn't work in the server as of 2022-09-30)
radio_button = widgets.Checkbox(value=False,
                                description='Radio Spectrum',
                                disabled=True,
                                indent=False
                                )

# The drop-drown for radio options
radio_drop_style = {'description_width': 'initial'}
radio_drop = widgets.Dropdown(options=radio_dict,
                              value=None,
                              description="Plot radio spectrum for:",
                              disabled=False,
                              style=radio_drop_style
                              )


def update_sensor_options(val):
    """
    this function updates the options in sensor_drop menu
    """
    sensor_drop.options = sensor_dict[spacecraft_drop.value]


def update_view_options(val):
    """
    updates the options and availability of view_drop menu
    """
    try:
        view_drop.disabled = False
        view_drop.options = view_dict[(spacecraft_drop.value, sensor_drop.value)]
        view_drop.value = view_drop.options[0]
    except KeyError:
        view_drop.disabled = True
        view_drop.value = None


def update_species_options(val):
    try:
        species_drop.options = species_dict[(spacecraft_drop.value, sensor_drop.value)]
    except KeyError:
        pass


def update_radio_options(val):
    radio_drop.disabled = not radio_button.value
    if radio_drop.disabled:
        radio_drop.value = None
    else:
        radio_drop.value = radio_drop.options[0]


def confirm_input(event_date: int, data_path: str, plot_path: str):

    print("You've chosen the following options:")
    print(f"Spacecraft: {spacecraft_drop.value}")
    print(f"Sensor: {sensor_drop.value}")
    print(f"Species: {species_drop.value}")
    print(f"Viewing: {view_drop.value}")
    print(f"Event_date: {event_date}")
    print(f"Data_path: {data_path}")
    print(f"Plot_path: {plot_path}")

    if spacecraft_drop.value == "Solar Orbiter":
        spacecraft_drop_value = "solo"
    elif spacecraft_drop.value == "STEREO-A":
        spacecraft_drop_value = "sta"
    elif spacecraft_drop.value == "STEREO-B":
        spacecraft_drop_value = "stb"
    else:
        spacecraft_drop_value = spacecraft_drop.value

    if sensor_drop.value in ["ERNE-HED"]:
        sensor_drop_value = "ERNE"
    else:
        sensor_drop_value = sensor_drop.value

    if species_drop.value == "protons":
        species_drop_value = 'p'
    else:
        species_drop_value = 'e'

    # this is to be fed into Event class as input
    global input_dict

    input_dict = {
        "Spacecraft": spacecraft_drop_value,
        "Sensor": sensor_drop_value,
        "Species": species_drop_value,
        "Viewing": view_drop.value,
        "Event_date": event_date,
        "Data_path": data_path,
        "Plot_path": plot_path
    }


# makes spacecraft_drop run these functions every time it is accessed by user
spacecraft_drop.observe(update_sensor_options)
spacecraft_drop.observe(update_view_options)
sensor_drop.observe(update_view_options)

# does the same but for sensor menu
spacecraft_drop.observe(update_species_options)
sensor_drop.observe(update_species_options)

# also observe the radio menu
# radio_button.observe(update_radio_options)
