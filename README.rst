SEPpy (ALPHA VERSION)
=====================

*This package is in alpha status! Intended for internal use only, as the syntax is in a floating state and the documentation is incomplete.*

**A compendium of Python data loaders for in-situ measurements of Solar Energetic Particles (SEP)**

So far combines the following loaders into one PyPI package:

- `psp-isois-loader <https://github.com/jgieseler/psp-isois-loader>`_
- `soho-loader <https://github.com/jgieseler/soho-loader>`_
- `solo-epd-loader <https://github.com/jgieseler/solo-epd-loader>`_ *
- `stereo-loader <https://github.com/jgieseler/stereo-loader>`_
- `wind-3dp-loader <https://github.com/jgieseler/wind-3dp-loader>`_

(* Note that `solo-epd-loader <https://github.com/jgieseler/solo-epd-loader>`_ is a `PyPI package itself <https://pypi.org/project/solo-epd-loader/>`_ that just is loaded here for completeness.)

Disclaimer
----------
This software is provided "as is", with no guarantee. It is no official data source, and not officially endorsed by the corresponding instrument teams. **Please always refer to the official data description of each instrument before using the data!**

Installation
------------

SEPpy requires python >= 3.6.

It can be installed from `PyPI <https://pypi.org/project/seppy/>`_ using:

.. code:: bash

    pip install seppy


Usage
-----

The standard usecase is to utilize the ``***_load`` function, which returns Pandas dataframe(s) of the corresponding measurements and a dictionary containing information on the energy channels. For example the SOHO/ERNE data from Apr 16 to Apr 20, 2021, can be obtained as follows:

.. code:: python

   from seppy.loader.soho import soho_load

   df, meta = soho_load(dataset="SOHO_ERNE-HED_L2-1MIN",
                        startdate="2021/04/16",
                        enddate="2021/04/20")

Note that the syntax is different for each loader! Please refer to the independent packages for more details and the correct useage:

- `psp-isois-loader <https://github.com/jgieseler/psp-isois-loader>`_
- `soho-loader <https://github.com/jgieseler/soho-loader>`_
- `solo-epd-loader <https://github.com/jgieseler/solo-epd-loader>`_
- `stereo-loader <https://github.com/jgieseler/stereo-loader>`_
- `wind-3dp-loader <https://github.com/jgieseler/wind-3dp-loader>`_
