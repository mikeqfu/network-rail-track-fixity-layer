"""
A class for preprocessing data related to *ballast*.
"""

import copy
import functools
import multiprocessing
import os
import warnings

import numpy as np
import pandas as pd
from pyhelpers._cache import _print_failure_msg
from pyhelpers.dirs import cd, cdd
from pyhelpers.ops import confirmed, merge_dicts
from pyhelpers.store import load_data, save_data, xlsx_to_csv
from pyrcs.converter import mile_yard_to_mileage

from src.utils.general import TrackFixityDB, add_sql_query_elr_condition, validate_column_names


class Ballast:
    """
    *Ballast*.

    This class focuses on preprocessing summary data specific to ballast. It provides
    methods to read, import and load ballast summary data from local directories or
    the project database, facilitating data cleaning, transformation and preparation
    for further analysis or modelling.

    .. note::

        This class currently handles with only a summary data about the ballasts.
    """

    #: Data name.
    NAME: str = 'Ballast'
    #: Pathname of a local directory where the ballast data is stored.
    DATA_DIR: str = os.path.relpath(cdd(NAME))
    #: Filename of the ballast summary data.
    SUM_FILENAME: str = "Ballast summary"

    #: Name of the schema for storing the ballast summary data.
    SCHEMA_NAME: str = copy.copy(NAME)
    #: Name of the table for storing the ballast summary data.
    SUM_TABLE_NAME: str = SUM_FILENAME.replace(' ', '_')

    def __init__(self, db_instance=None):
        """
        :param db_instance: Optional PostgreSQL database instance; defaults to ``None``.
        :type db_instance: TrackFixityDB | None

        :ivar dict object_dtype: Object field types in the ballast summary data.
        :ivar dict float_dtype: Float field types in the ballast summary data.
        :ivar dict int_dtype: Integer field types in the ballast summary data.
        :ivar dict var_dtype: Variable/mixed field types in the ballast summary data.
        :ivar dict dtypes: Field types of the ballast summary data.

        :ivar TrackFixityDB db_instance: PostgreSQL database instance used for
            data retrieval and storage.

        **Examples**::

            >>> from src.preprocessor import Ballast
            >>> blst = Ballast()
            >>> blst.NAME
            'Ballast summary'

        .. figure:: ../_images/ballast_schema.*
            :name: ballast_schema
            :align: center
            :width: 100%

            Snapshot of the *Ballast* schema.
        """

        # noinspection SpellCheckingInspection
        self.object_dtype = {
            'ID': str,
            'Track priority': str,
            'Operating route': str,
            'SRS': str,
            'ELR': str,
            'IMDM': str,
            'TME': str,
            'TSM': str,
            'Route class': str,
            'Electricification': str,  # Electrification?
            'Embankment': str,
            'Soil cutting': str,
            'Rock cutting': str,
            'Station': str,
            'Tunnel': str,
            'Track': str,
            'Track type': str,
            'Rail type': str,
            'Rail alloy': str,
            'Rail serviceable': str,
            'Sleeper type': str,
            'Sleeper serviceable': str,
            'Baseplate': str,
            'Fastening': str,
            'Fixing': str,
            'Last ballast renewal': str,
            'Switch type': str,
            'Vertical inclined': str,
            'Switch Name': str,
            'Switch joint': str,
            'Switch blade': str,
            'Track construction band': str,
        }
        self.float_dtype = {
            'Curvature': np.float64,
            'Cant': np.float64,
            'EMGTPA (2012)': np.float64,
            'Rail used life fraction': np.float64,
            'Sleeper used life fraction': np.float64,
            'Ballast fouling index': np.float64,
            'Switch used life fraction': np.float64,
            'LTSF': np.float64,
            'BCF': np.float64,
        }
        self.int_dtype = {
            'GEOGIS Switch ID': 'Int64',
            'TID': 'Int64',
            'Start Mile': 'Int64',
            'Start Yard': 'Int64',
            'End Mile': 'Int64',
            'End Yard': 'Int64',
            'Max speed': 'Int64',
            'Max axle load': 'Int64',
            'Left rail year': 'Int64',
            'Right rail year': 'Int64',
            'Sleeper year': 'Int64',
            'Ballast year': 'Int64',
            'MGTPA (2012)': 'Int64',
            'Rail cumulative EMGT': 'Int64',
            'Sleeper cumulative EMGT': 'Int64',
            'Ballast cumulative EMGT': 'Int64',
        }
        self.var_dtype = {
            'Track category (2012)': str  # 1-6, 1A
        }

        self.dtypes = merge_dicts(self.object_dtype, self.float_dtype, self.int_dtype, self.var_dtype)

        self.db_instance = db_instance

    def read_data(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read ballast summary data from a local directory.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: DataFrame containing the ballast summary data, or ``None`` if data cannot be read.
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor import Ballast
            >>> blst = Ballast()
            >>> data = blst.read_data()
            >>> data.shape
            (650867, 58)
        """

        path_to_pickle = cd(self.DATA_DIR, self.SUM_FILENAME + ".pkl")

        if os.path.isfile(path_to_pickle) and not update:
            data = load_data(path_to_pickle)

        else:
            if verbose:
                print(f"Parsing data of {self.NAME.lower()}", end=" ... ")

            try:
                path_to_file = cd(self.DATA_DIR, self.SUM_FILENAME + ".xlsx")
                # summary = pd.read_excel(path_to_file, dtype=self.DataTypes)

                path_to_csv_temp = xlsx_to_csv(path_to_file)
                data = pd.read_csv(path_to_csv_temp, dtype=self.dtypes, low_memory=False)
                os.remove(path_to_csv_temp)

                str_col = list(self.object_dtype.keys())
                data.replace(dict(zip(str_col, [{np.nan: ''}] * len(str_col))), inplace=True)

                if verbose:
                    print("Done.")

                save_data(data, path_to_file=path_to_pickle, verbose=verbose)

            except Exception as e:
                _print_failure_msg(e=e, msg="Failed.")
                data = None

        return data

    def import_data(self, update=False, confirmation_required=True, verbose=True, **kwargs):
        """
        Import ballast summary data into the project database.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param confirmation_required: Whether a confirmation is required to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the
            `pyhelpers.dbms.PostgreSQL.import_data`_ method.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.preprocessor import Ballast
            >>> blst = Ballast()
            >>> blst.import_data(if_exists='replace')
            To import data of ballast summary into the table "Ballast"."Ballast_summary"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/ballast_summary_tbl.*
            :name: ballast_summary_tbl
            :align: center
            :width: 100%

            Snapshot of the "Ballast"."Ballast_summary" table.
        """

        dat_name = f"data of {self.NAME.lower()}"
        tbl_name = f'"{self.SCHEMA_NAME}"."{self.SUM_TABLE_NAME}"'

        if confirmed(f"To import {dat_name} into the table {tbl_name}?\n", confirmation_required):

            ballast_summary = self.read_data(update=update, verbose=verbose)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing the data", end=" ... ")

            try:
                self.db_instance.import_data(
                    data=ballast_summary, schema_name=self.SCHEMA_NAME,
                    table_name=self.SUM_TABLE_NAME, method=self.db_instance.psql_insert_copy,
                    confirmation_required=False, **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e=e, msg="Failed.")

    @staticmethod
    def _nr_mileage(mi_yd_list):
        mileage = mi_yd_list.apply(lambda x: mile_yard_to_mileage(x.iloc[0], x.iloc[1]), axis=1)
        return mileage

    def load_data(self, elr=None, column_names=None, fmt_nr_mileage=True, **kwargs):
        # noinspection PyShadowingNames
        """
        Load ballast summary data from the project database.

        :param elr: Engineer's Line Reference (ELR); defaults to ``None``.
        :type elr: str | list | tuple | None
        :param column_names: Names of columns (a subset) to be queried; defaults to ``None``.
        :type column_names: str | list | None
        :param fmt_nr_mileage: Whether to format Network Rail mileage data; defaults to ``True``.
        :type fmt_nr_mileage: bool
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.read_sql_query`_.
        :return: Dictionary containing the loaded ballast summary data.
        :rtype: pandas.DataFrame

        .. _`pyhelpers.dbms.PostgreSQL.read_sql_query`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.read_sql_query.html

        **Examples**::

            >>> from src.preprocessor import Ballast
            >>> blst = Ballast()
            >>> elr = ['ECM7', 'ECM8']
            >>> data_ecm7_8 = blst.load_data(elr, fmt_nr_mileage=False)
            >>> data_ecm7_8.shape
            (8184, 58)
            >>> data_ecm7_8_ = blst.load_data(elr, fmt_nr_mileage=True)
            >>> data_ecm7_8_.shape
            (8184, 60)
            >>> data_ecm7_8_[['StartMileage', 'EndMileage']].head()
               StartMileage  EndMileage
            0        0.0000      0.0054
            1        0.0054      0.0060
            2        0.0060      0.0066
            3        0.0066      0.0084
            4        0.0084      0.0092
        """

        essential_columns = ['ELR', 'Start Mile', 'Start Yard', 'End Mile', 'End Yard']
        if column_names is not None:
            assert isinstance(column_names, (list, tuple))
            assert all(col in essential_columns for col in column_names)

        column_names_ = validate_column_names(column_names=column_names)
        sql_query = f'SELECT {column_names_} FROM "{self.SCHEMA_NAME}"."{self.SUM_TABLE_NAME}"'

        sql_query = add_sql_query_elr_condition(sql_query, elr=elr)

        if column_names_ == '*' or 'ID' in column_names_:
            sort_by_cols = ['ID']
        else:
            sort_by_cols = essential_columns.copy()

        if self.db_instance is None:
            self.db_instance = TrackFixityDB()

        data = self.db_instance.read_sql_query(
            sql_query=sql_query, low_memory=False, **kwargs).sort_values(
            sort_by_cols, ignore_index=True)

        if fmt_nr_mileage:
            n = multiprocessing.cpu_count()

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                start_mi_yd_list, end_mi_yd_list = map(
                    functools.partial(np.array_split, indices_or_sections=n),
                    [data[['Start Mile', 'Start Yard']], data[['End Mile', 'End Yard']]])

            with multiprocessing.Pool(processes=n) as p:
                start_mileage = p.map(self._nr_mileage, start_mi_yd_list)
                end_mileage = p.map(self._nr_mileage, end_mi_yd_list)

            data['StartMileage'], data['EndMileage'] = map(pd.concat, [start_mileage, end_mileage])

        return data


# if __name__ == '__main__':
#     blst = Ballast()
#     ballast_summary = blst.read_data(update=True, verbose=True)
#     blst.import_data(if_exists='replace', confirmation_required=False, verbose=True)
#     ballast_summary_ecm7_8 = blst.load_data(elr=['ECM7', 'ECM8'], fmt_nr_mileage=True)
