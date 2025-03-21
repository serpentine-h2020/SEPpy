seppy
=====

|pypi Version| |python version| |pytest| |codecov| |repostatus| |zenodo doi|

.. |pypi Version| image:: https://img.shields.io/pypi/v/seppy?style=flat&logo=pypi
   :target: https://pypi.org/project/seppy/
.. |python version| image:: https://img.shields.io/pypi/pyversions/seppy?style=flat&logo=python
.. |zenodo doi| image:: https://zenodo.org/badge/451799504.svg
   :target: https://zenodo.org/badge/latestdoi/451799504
.. |pytest| image:: https://github.com/serpentine-h2020/SEPpy/actions/workflows/pytest.yml/badge.svg?branch=main
.. |codecov| image:: https://codecov.io/gh/serpentine-h2020/SEPpy/branch/main/graph/badge.svg?token=FYELM4Y7DF 
   :target: https://codecov.io/gh/serpentine-h2020/SEPpy
.. |repostatus| image:: https://www.repostatus.org/badges/latest/active.svg
   :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.
   :target: https://www.repostatus.org/#active

*This package is in development status! Intended for internal use only, as the syntax is in a floating state and the documentation is incomplete.*

**A compendium of Python data loaders and analysis tools for in-situ measurements of Solar Energetic Particles (SEP)**

So far combines loaders for the following instruments into one PyPI package:

- Parker Solar Probe: ISOIS
- SOHO: CELIAS, COSTEP-EPHIN, ERNE
- Solar Orbiter: EPD (STEP, EPT, HET)*, MAG
- STEREO: HET, LET, SEPT, MAG
- Wind: 3DP

(* Note that `solo-epd-loader <https://github.com/jgieseler/solo-epd-loader>`_ is a `PyPI package itself <https://pypi.org/project/solo-epd-loader/>`_ that just is loaded here for completeness.)


Disclaimer
----------
This software is provided "as is", with no guarantee. It is no official data source, and not officially endorsed by the corresponding instrument teams. **Please always refer to the official data description of each instrument before using the data!**

Installation
------------

seppy requires python >= 3.9.

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

Note that the syntax is different for each loader! `Please refer to this Notebook for more info and examples for the different data sets! <https://github.com/jgieseler/serpentine/blob/main/notebooks/sep_analysis_tools/data_loader.ipynb>`_




Citation
--------

Please cite the following paper if you use **seppy** in your publication:

Palmroos, C., Gieseler, J., Dresing, N., Morosan, D.E., Asvestari, E., Yli-Laurila, A., Price, D.J., Valkila, S., Vainio, R. (2022).
Solar Energetic Particle Time Series Analysis with Python. *Front. Astronomy Space Sci.* 9. `doi:10.3389/fspas.2022.1073578 <https://doi.org/10.3389/fspas.2022.1073578>`_ 
