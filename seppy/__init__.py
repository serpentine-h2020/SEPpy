
from .version import version as __version__
from seppy.util import custom_warning

# __all__ = []  # defines which functions, variables etc. will be loaded when running "from pyonset import *"

custom_warning('Breaking changes in SEPpy v0.4.0: The metadata for SOHO/EPHIN, SOHO/ERNE, STEREO/SEPT, and Wind/3DP have changed! See https://github.com/serpentine-h2020/SEPpy/releases/tag/v0.4.0 for details. (You can ignore this if you do not invoke SEPpy manually.)')
