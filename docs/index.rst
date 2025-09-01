###################
SEPpy documentation
###################

**A compendium of Python data loaders and analysis tools for in-situ measurements of Solar Energetic Particles (SEP)**

So far combines loaders for the following instruments into one PyPI package:

* BepiColombo: SIXS-P
* Parker Solar Probe: ISOIS
* SOHO: CELIAS, COSTEP-EPHIN, ERNE
* Solar Orbiter: EPD [1]_ (STEP, EPT, HET), MAG
* STEREO: HET, LET, SEPT, MAG
* Wind: 3DP

.. [1] Note that `solo-epd-loader <https://github.com/jgieseler/solo-epd-loader>`_ is an independent `PyPI package <https://pypi.org/project/solo-epd-loader/>`_ that is loaded here for completeness.

.. note::
  Please cite the following paper if you use **SEPpy** in your publication:

  Palmroos, C., Gieseler, J., Dresing, N., Morosan, D.E., Asvestari, E., Yli-Laurila, A., Price, D.J., Valkila, S., Vainio, R. (2022).
  Solar Energetic Particle Time Series Analysis with Python. *Front. Astronomy Space Sci.* 9. `doi:10.3389/fspas.2022.1073578 <https://doi.org/10.3389/fspas.2022.1073578>`_


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   usage
   api
   genindex
