"""
A class for preprocessing data of *geology*.
"""

import copy
import os

import numpy as np
import pandas as pd
from pyhelpers._cache import _print_failure_msg
from pyhelpers.dirs import cd, cdd
from pyhelpers.ops import confirmed
from pyhelpers.store import load_data, save_data

from src.utils.general import TrackFixityDB, add_sql_query_elr_condition


class Geology:
    """
    *Geology*.

    .. note::

        This class currently handles only a summary about the geology.
    """

    #: Data name.
    NAME: str = 'Geology'
    #: Pathname of a local directory where the geology data is stored.
    DATA_DIR: str = os.path.relpath(cdd(NAME))
    #: Filename of the original geology data file.
    GEOL_FILENAME: str = 'Geology'

    #: Name of schema for storing the geology data.
    SCHEMA_NAME: str = copy.copy(NAME)
    #: Name of the table for storing the geology data.
    GEOL_TABLE_NAME: str = copy.copy(GEOL_FILENAME)

    def __init__(self, db_instance=None):
        """
        :param db_instance: PostgreSQL database instance; defaults to ``None``.
        :type db_instance: TrackFixityDB | None

        :ivar dict dtypes: Field types of the geology data.
        :ivar TrackFixityDB db_instance: PostgreSQL database instance used for operations.

        **Examples**::

            >>> from src.preprocessor import Geology
            >>> geol = Geology()
            >>> geol.NAME
            'Geology'

        .. figure:: ../_images/geology_schema.*
            :name: geology_schema
            :align: center
            :width: 100%

            Snapshot of the *Geology* schema.
        """

        self.dtypes = {
            'Layer': str,
            'ELR': str,
            'Start': str,
            'End': str,
            'LEX_D': str,
            'ROCK_D': str
        }

        self.db_instance = db_instance

    def read_summary(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read geology data from a local directory.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of geology.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import Geology
            >>> geol = Geology()
            >>> data = geol.read_summary()
            >>> data.shape
            (55642, 6)
        """

        path_to_pickle = cd(self.DATA_DIR, self.GEOL_FILENAME + ".pkl")

        if os.path.isfile(path_to_pickle) and not update:
            data = load_data(path_to_pickle)

        else:
            data = pd.read_csv(path_to_pickle.replace(".pkl", ".csv"), dtype=self.dtypes)

            data.replace({np.nan: ''}, inplace=True)

            save_data(data, path_to_pickle, verbose=verbose)

        return data

    def import_summary(self, update=False, confirmation_required=True, verbose=True, **kwargs):
        """
        Import geology data into the project database.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param confirmation_required: Whether confirmation is required to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.import_data`_.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/
            _generated/pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.preprocessor import Geology
            >>> geol = Geology()
            >>> geol.import_summary(if_exists='replace')
            To import data of geology into the table "Geology"."Geology"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/geology_geology_tbl.*
            :name: geology_geology_tbl
            :align: center
            :width: 100%

            Snapshot of the "Geology"."Geology" table.
        """

        dat_name = f"data of {self.NAME.lower()}"
        tbl_name = f"\"{self.SCHEMA_NAME}\".\"{self.NAME}\""

        if confirmed(f"To import {dat_name} into the table {tbl_name}?\n", confirmation_required):
            data = self.read_summary(update=update, verbose=verbose)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing the data", end=" ... ")

            try:
                self.db_instance.import_data(
                    data=data, schema_name=self.SCHEMA_NAME, table_name=self.GEOL_TABLE_NAME,
                    method=self.db_instance.psql_insert_copy, confirmation_required=False,
                    **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")

    def load_summary(self, elr=None, **kwargs):
        # noinspection PyShadowingNames
        """
        Load geology data from the project database.

        :param elr: Engineer's Line Reference(s); defaults to ``None``.
        :type elr: str | list | tuple | None
        :param kwargs: [Optional] parameters for the method
            `pyhelpers.dbms.PostgreSQL.read_sql_query`_.
        :return: Data of geology.
        :rtype: pandas.DataFrame

        .. _`pyhelpers.dbms.PostgreSQL.read_sql_query`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.read_sql_query.html

        **Examples**::

            >>> from src.preprocessor import Geology
            >>> geol = Geology()
            >>> data = geol.load_summary(elr=['ECM7', 'ECM8'])
            >>> data.shape
            (662, 6)
        """

        if self.db_instance is None:
            self.db_instance = TrackFixityDB()

        sql_query = f'SELECT * FROM "{self.SCHEMA_NAME}"."{self.GEOL_TABLE_NAME}"'

        sql_query = add_sql_query_elr_condition(sql_query, elr=elr)
        sql_query += 'ORDER BY "ELR" ASC, "Start" ASC, "End" ASC'

        data = self.db_instance.read_sql_query(sql_query=sql_query, **kwargs)

        return data
