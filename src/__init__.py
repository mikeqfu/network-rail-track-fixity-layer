"""Python code for the project 'Track Fixity Layer'"""

import datetime
import json
import pkgutil

from pyhelpers.settings import mpl_preferences

from . import modeller, preprocessor, shaft, utils

metadata = json.loads(pkgutil.get_data(__name__, "data/metadata").decode())

__project__ = metadata['Project']
__version__ = metadata['Version']
__desc__ = metadata['Description']
__authors__ = metadata['Authors']
__affil__ = metadata['Affiliation']
__copyright__ = f'2021-{datetime.datetime.now().year}, {__authors__}'

__all__ = ['preprocessor', 'shaft', 'modeller', 'utils']

mpl_preferences(font_name='Times New Roman')
