"""
Further processing and exploration of the data that are preprocessed by the subpackage
:mod:`~src.preprocessor` to generate a comprehensive data set for the development of a machine
learning model on track fixity.

The data processing includes three tasks:

    - Calculation of track movement.
    - Collation of the data of variables that influence the track movement.
    - Integration of data sets of the calculated track movement and the influencing variables.
"""

from .feature import FeatureCollator
from .krdz_gear import KRDZGear
from .movement import TrackMovement
from .pcd_handler import PCDHandler

__all__ = ['PCDHandler', 'KRDZGear', 'TrackMovement', 'FeatureCollator']
