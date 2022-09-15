# SEPpy

**A compendium of Python data loaders for in-situ measurements of Solar Energetic Particles (SEP)**

So far combines the following loaders into one PyPI package:

- [psp-isois-loader](https://github.com/jgieseler/psp-isois-loader)
- [soho-loader](https://github.com/jgieseler/soho-loader)
- [solo-epd-loader](https://github.com/jgieseler/solo-epd-loader)*
- [stereo-loader](https://github.com/jgieseler/stereo-loader)
- [wind-3dp-loader](https://github.com/jgieseler/wind-3dp-loader)

*Note that [solo-epd-loader](https://github.com/jgieseler/solo-epd-loader) is a [PyPI package itself](https://pypi.org/project/solo-epd-loader/) that just is loaded here for completeness.

Disclaimer
----------
This software is provided "as is", with no guarantee. It is no official data source, and not officially endorsed by the corresponding instrument teams. **Please always refer to the official data description of each instrument before using the data!**

Installation
------------

solo_epd_loader requires python >= 3.6.

It can be installed from `PyPI <https://pypi.org/project/seppy/>`_ using:

.. code:: bash

    pip install seppy


Usage
-----

The standard usecase is to utilize the ``epd_load`` function, which
returns Pandas dataframe(s) of the EPD measurements and a dictionary
containing information on the energy channels.

.. code:: python

   from solo_epd_loader import epd_load

   df_1, df_2, energies = \
       epd_load(sensor, level, startdate, enddate=None, viewing=None, path=None, autodownload=False)
