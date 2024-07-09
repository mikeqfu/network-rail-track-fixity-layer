"""
A class for preprocessing data from *Integrated Network Model (INM)*.
"""

import copy
import os

import numpy as np
import pandas as pd
from pyhelpers._cache import _print_failure_msg
from pyhelpers.dirs import cd, cdd
from pyhelpers.ops import confirmed, merge_dicts
from pyhelpers.store import load_data, save_data
from pyhelpers.text import get_acronym
from pyrcs.converter import fix_mileage

from src.utils.general import TrackFixityDB, add_sql_query_elr_condition


class INM:
    """
    *Integrated Network Model*.

    .. note::

        This class currently handles only the INM combined data report, which is available
        as a CSV file.
    """

    #: Data name.
    NAME: str = 'Integrated Network Model'
    #: Acronym for the data name.
    ACRONYM: str = get_acronym(NAME)
    #: Pathname of a local directory where the INM data is stored.
    DATA_DIR: str = os.path.relpath(cdd(ACRONYM))

    #: Name of the INM combined data report (CDR).
    CDR_NAME: str = "Combined data report"
    #: Filename of the original INM CDR data.
    CDR_FILENAME: str = f"{ACRONYM} {CDR_NAME.lower()}"

    #: Name of schema for storing the INM data.
    SCHEMA_NAME: str = copy.copy(ACRONYM)
    #: Name of table for storing the data of INM combined data report.
    CDR_TABLE_NAME: str = copy.copy(CDR_NAME)

    def __init__(self, db_instance=None):
        """
        :param db_instance: PostgreSQL database instance; defaults to ``None``.
        :type db_instance: TrackFixityDB | None

        :ivar dict cdr_dtypes_object: Field types for object data in the INM CDR.
        :ivar dict cdr_dtypes_float: Field types for float data in the INM CDR.
        :ivar dict cdr_dtypes_int: Field types for integer data in the INM CDR.
        :ivar dict cdr_dtypes_mix: Field types for mixed-type data in the INM CDR,
            potentially containing errors.
        :ivar dict cdr_dtypes: Field types of the INM CDR data.
        :ivar dict mileage_dtypes: Field types for mileage data in the INM CDR.

        :ivar TrackFixityDB db_instance: PostgreSQL database instance for database operations.

        **Examples**::

            >>> from src.preprocessor import INM
            >>> inm = INM()
            >>> inm.NAME
            'Integrated Network Model'

        .. figure:: ../_images/inm_schema.*
            :name: inm_schema
            :align: center
            :width: 100%

            Snapshot of the *INM* schema.
        """

        self.cdr_dtypes_object = {
            'RTE_NAME': str,
            'RTE_ORG_CODE': str,
            'MDU_NAME': str,
            'MDU_ORG_CODE': str,
            'TME_NAME': str,
            'TME_ORG_CODE': str,
            'TSM_NAME': str,
            'TSM_ORG_CODE': str,
            'ELR': str,
            'ELR_STARTMEASURE': str,
            'ELR_ENDMEASURE': str,
            'TRACKTYPE': str,
            'TRACKSECTION': str,
            'LEFTRAILWEIGHT': str,
            'LEFTRAILALLOY': str,
            'LEFTRAILNEWSERVICEABLE': str,
            'RIGHTRAILWEIGHT': str,
            'RIGHTRAILALLOY': str,
            'RIGHTRAILNEWSERVICEABLE': str,
            'CHECKRAIL': str,
            'CONDUCTORRAIL': str,
            'ELECTRIFICATIONTYPE': str,
            'CONDITIONS': str,
            'SLEEPER': str,
            'SLEEPERBASEPLATETYPE': str,
            'FIXING': str,
            'FASTENING': str,
            'SLEEPERSPER60FOOTLENGTH': str,
            'SLEEPERNEWSERVICEABLE': str,
            'PATCHPROGRAM': str,
            'BALLASTMETHOD': str,
            'TAMPING': str,
            'TRACK_CATEGORY': str,
            'SMOOTH_TRACK_CAT': str,
            'OLD_TRACK_CATEGORY': str,
            'RELAYINGPOLICY': str,
        }
        self.cdr_dtypes_float = {
            'ANNUAL_TONNAGE': np.float64,
            'EMGTPA': np.float64,
            'MAX_AXLE_WEIGHT': np.float64,
        }
        self.cdr_dtypes_int = {
            'LEFTRAILYEARLAID': 'Int64',
            'LEFTRAILYEARROLLED': 'Int64',
            'RIGHTRAILYEARLAID': 'Int64',
            'RIGHTRAILYEARROLLED': 'Int64',
            'SLEEPERYEAR': 'Int64',
            'PATCHYEAR': 'Int64',
            'BALLASTYEAR': 'Int64',
            'TRACK_PRIORITY': 'Int64',
            'TRACK_POSITION': 'Int64',
            'SPEED_LEVEL_RAISED': 'Int64',
            'SPEED_LEVEL_REVERSE_DIRECTION': 'Int64',
            'OVERRIDE': 'Int64',
            'ELR_ENDMEASURE_YARD': 'Int64',
            'ELR_STARTMEASURE_YARD': 'Int64',
            'REF_TRACKID': 'Int64',
            'SPEED_LEVEL_NORMAL': 'Int64',
        }

        self.cdr_dtypes_mix = {
            'SLEEPERSPER60FOOTLENGTH': str,
            'PATCHPROGRAM': str,
            'TRACK_CATEGORY': str,
            'SMOOTH_TRACK_CAT': str}

        self.cdr_dtypes = merge_dicts(
            self.cdr_dtypes_object, self.cdr_dtypes_float, self.cdr_dtypes_int, self.cdr_dtypes_mix)

        self.mileage_dtypes = {'ELR_STARTMEASURE': str, 'ELR_ENDMEASURE': str}

        self.db_instance = db_instance

    def read_combined_data_report(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read INM combined data report from a local directory.

        :param update: Whether to re-read the original INM combined data report file;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of the INM combined data report.
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor import INM
            >>> inm = INM()
            >>> inm_cdr = inm.read_combined_data_report()
            >>> inm_cdr.shape
            (486674, 55)

        .. note::

            - ``'SLEEPERSPER60FOOTLENGTH'`` (``int``): strings
              (i.e. ``'PAN'`` and ``'NONE'``) and ``nan``
            - ``'PATCHPROGRAM'`` (``str`` of 2 digits of int): 2019, ``'Patch Program'`` and ``nan``
            - ``'TRACK_CATEGORY'`` (``int``, 1-6): ``nan`` and ``'1A'``
            - ``'SMOOTH_TRACK_CAT'`` (``int``, 1-6): ``nan`` and ``'1A'``
        """

        path_to_pickle = cd(self.DATA_DIR, self.CDR_FILENAME + ".pkl")

        if os.path.isfile(path_to_pickle) and not update:
            inm_cdr = load_data(path_to_pickle)

        else:
            if verbose:
                print(f"Parsing {self.CDR_FILENAME}", end=" ... ")

            try:
                path_to_csv = path_to_pickle.replace(".pkl", ".csv")
                inm_cdr = pd.read_csv(path_to_csv, dtype=self.cdr_dtypes)

                mileage_col = list(self.mileage_dtypes.keys())
                inm_cdr[mileage_col] = inm_cdr[mileage_col].map(lambda x: fix_mileage(x.strip()))

                str_col = list(self.cdr_dtypes_object.keys())
                inm_cdr.replace(dict(zip(str_col, [{np.nan: ''}] * len(str_col))), inplace=True)

                if verbose:
                    print("Done.")

                save_data(inm_cdr, path_to_pickle, verbose=verbose)

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")
                inm_cdr = None

        return inm_cdr

    def import_combined_data_report(self, update=False, confirmation_required=True, verbose=True,
                                    **kwargs):
        """
        Import the data from the Integrated Network Model (INM) combined data report
        into the project database.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param confirmation_required: Whether to ask for confirmation before proceeding;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters to pass to the method
            `pyhelpers.dbms.PostgreSQL.import_data`_.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.preprocessor import INM
            >>> inm = INM()
            >>> inm.import_combined_data_report()
            To import INM combined data report into the table "INM"."Combined data report"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/inm_cdr_tbl.*
            :name: inm_cdr_tbl
            :align: center
            :width: 100%

            Snapshot of the "INM"."Combined data report" table.
        """

        dat_name = f"{self.ACRONYM} {self.CDR_NAME.lower()}"
        tbl_name = f"\"{self.SCHEMA_NAME}\".\"{self.CDR_NAME}\""

        if confirmed(f"To import {dat_name} into the table {tbl_name}?\n", confirmation_required):

            inm_cdr = self.read_combined_data_report(update=update, verbose=verbose)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing the data", end=" ... ")

            try:
                self.db_instance.import_data(
                    data=inm_cdr, schema_name=self.SCHEMA_NAME, table_name=self.CDR_NAME,
                    method=self.db_instance.psql_insert_copy, confirmation_required=False,
                    **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")

    def load_combined_data_report(self, elr=None, **kwargs):
        # noinspection PyShadowingNames
        """
        Load the data of the Integrated Network Model (INM) combined data report
        from the project database.

        :param elr: Engineer's Line Reference(s) to filter the data; defaults to ``None``.
        :type elr: str | list | tuple | None
        :param kwargs: [Optional] additional parameters to pass to the method
            `pyhelpers.dbms.PostgreSQL.read_sql_query`_.
        :return: Data of the INM combined data report.
        :rtype: pandas.DataFrame

        .. _`pyhelpers.dbms.PostgreSQL.read_sql_query`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.read_sql_query.html

        **Examples**::

            >>> from src.preprocessor import INM
            >>> inm = INM()
            >>> inm_cdr = inm.load_combined_data_report(elr=['ECM7', 'ECM8'])
            >>> inm_cdr.shape
            (5510, 55)
        """

        if self.db_instance is None:
            self.db_instance = TrackFixityDB()

        sql_query = f'SELECT * FROM "{self.SCHEMA_NAME}"."{self.CDR_TABLE_NAME}"'

        sql_query = add_sql_query_elr_condition(sql_query, elr=elr)
        sql_query += 'ORDER BY "ELR" ASC, "ELR_STARTMEASURE_YARD" ASC, "ELR_ENDMEASURE_YARD" ASC'

        inm_cdr = self.db_instance.read_sql_query(sql_query=sql_query, **kwargs)

        return inm_cdr
