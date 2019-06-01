"""
adaptive_md_tools
This package provides tools for analyizing adaptive partitioning simulations and
 calculating the an excess proton
"""

# Add imports here
from .AdaptiveMD import *


# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
