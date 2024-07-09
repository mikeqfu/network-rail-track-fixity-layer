"""
A class for preprocessing data about *track layout and quality*.
"""

import copy
import datetime
import itertools
import os

import natsort
import numpy as np
import pandas as pd
import shapely.ops
import shapely.wkt
from pydriosm.reader import SHPReadParse
from pyhelpers._cache import _print_failure_msg
from pyhelpers.dirs import cd, cdd
from pyhelpers.ops import confirmed
from pyhelpers.store import load_data, save_data
from pyrcs.converter import mileage_str_to_num, mileage_to_yard, yard_to_mileage

from src.utils.general import (TrackFixityDB, add_sql_query_date_condition,
                               add_sql_query_elr_condition)


def make_dtf_natsort_key(dtf_filename):
    """
    Create a sorting key for each track quality file (.dtf).

    This key is used to determine the order of raw files when listing them in a data folder.

    :param dtf_filename: Name of a .dtf file.
    :type dtf_filename: str
    :return: A track ID and date extracted from the filename.
    :rtype: tuple
    """

    if os.path.isfile(dtf_filename):
        dtf_filename = os.path.basename(dtf_filename)

    track_id, file_date = dtf_filename.split('.')[0].split('_')
    file_date = datetime.datetime.strptime(file_date, '%d-%m-%Y')
    # ns_key = (track_id, file_date)

    return track_id, file_date


def get_dtf_dates(dtf_file_paths):
    """
    Extract data dates from .dtf filenames based on their local pathnames.

    :param dtf_file_paths: Pathnames of the .dtf files.
    :type dtf_file_paths: list
    :return: Dictionary mapping .dtf filenames to their extracted data dates.
    :rtype: dict
    """

    temp = [os.path.basename(x).split('_') for x in dtf_file_paths]
    data_dates_ = [{k: v.split('.')[0]} for k, v in temp if v.endswith(".dtf")]

    data_dates = {k: [d.get(k) for d in data_dates_ if d.get(k)] for k in set().union(*data_dates_)}

    return data_dates


class Track:
    """
    *Track layout and quality*.

    This class currently handles:

        - (calibrated) shapefile of the track layout (including reference lines and changes)
        - track quality files
    """

    #: Name of the data.
    NAME: str = 'Track'
    #: Pathname of the local directory where the track data are stored.
    DATA_DIR: str = os.path.relpath(cdd(NAME))
    #: Pathname of the local directory where the track shapefiles are stored.
    SHP_DIR_PATH: str = os.path.relpath(cd(DATA_DIR, "Shapefiles"))
    #: Filename of track shapefile (calibrated) data.
    TRK_SHP_FILENAME: str = "tracks_whole_uk_April2020_Calibrated"
    #: Short description of track shapefile (calibrated) data.
    TRK_SHP_DESC: str = 'Shapefile (calibrated) of tracks of the whole UK'

    #: Filename of Network Model gauging changes data.
    NM_CHANGES_FILENAME: str = "NM_changes_39pt1_to_39pt2_gauging"
    #: Short description of Network Model gauging changes data.
    NM_CHANGES_DESC: str = "Network Model gauging changes (39pt1 to 39pt2)"

    #: Filename of reference line shapefile (calibrated) data.
    REF_LINE_FILENAME: str = "Calibrated_ReferenceLine"
    #: Short description of reference line shapefile (calibrated) data.
    REF_LINE_DESC: str = "Shapefile (calibrated) of reference line"

    #: Pathname of the local directory where the track quality files are stored.
    DTF_DIR_PATH: str = os.path.relpath(cd(DATA_DIR, "Track quality"))
    #: Key of the parsed track quality data (in ``dict`` format).
    DTF_KEY: str = 'TQ'

    #: Name of the schema for storing the track data.
    SCHEMA_NAME: str = copy.copy(NAME)
    #: Name of the table storing the track shapefile (calibrated) data.
    TRK_SHP_TABLE_NAME: str = 'Tracks_shapefile_calibrated'
    #: Name of the table storing reference line shapefile (calibrated) data.
    REF_LINE_TABLE_NAME: str = 'RefLine_shapefile_calibrated'
    #: Name of the table storing the track quality files.
    DTF_TABLE_NAME: str = 'Track_quality'
    #: Name of the table storing pseudo mileages.
    PSEUDO_MILEAGE_TABLE_NAME: str = 'Pseudo_mileage'

    def __init__(self, db_instance=None):
        # noinspection PyShadowingNames
        """
        :param db_instance: PostgreSQL database instance; defaults to ``None``.
        :type db_instance: TrackFixityDB | None

        :ivar list shp_data_dates: Dates of the track shapefiles.
        :ivar list track_quality_tid: Track IDs for the track quality files.
        :ivar list dtf_pathnames: File paths to the track quality files.
        :ivar list dtf_data_dates: Dates of the track quality files.
        :ivar pyhelpers.dbms.PostgreSQL db_instance: PostgreSQL database instance.

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> trk.NAME
            'Track'

        .. figure:: ../_images/trk_schema.*
            :name: trk_schema
            :align: center
            :width: 100%

            Snapshot of the *Track* schema.
        """

        self.shp_data_dates = [
            x for x in os.listdir(self.SHP_DIR_PATH) if os.path.isdir(cd(self.SHP_DIR_PATH, x))]

        self.track_quality_tid = [
            x for x in os.listdir(cd(self.DTF_DIR_PATH)) if os.path.isdir(cd(self.DTF_DIR_PATH, x))]

        # noinspection PyTypeChecker
        self.dtf_pathnames = natsort.natsorted(
            [cd(rt, n)
             for rt, _, fs in os.walk(self.DTF_DIR_PATH) for n in fs if n.endswith('.dtf')],
            key=make_dtf_natsort_key)

        self.dtf_data_dates = get_dtf_dates(self.dtf_pathnames)

        self.db_instance = db_instance

    # == Shapefiles ================================================================================

    def read_gauging_changes(self, trk_date, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read data of network model changes on '39pt1_to_39pt2_gauging' from a local directory.

        :param trk_date: Data date of track shapefiles.
        :type trk_date: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of the network model changes on '39pt1_to_39pt2_gauging'.
        :rtype: dict

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> trk_date = '202004'
            >>> gauging_changes = trk.read_gauging_changes(trk_date)
            >>> # gauging_changes = trk.read_gauging_changes(trk_date, update=True, verbose=True)
            >>> type(gauging_changes)
            dict
            >>> list(gauging_changes.keys())
            ['Links_added', 'Links_deleted', 'Nodes_added', 'Nodes_deleted']
            >>> gauging_changes['Links_added'].shape
            (3, 11)
            >>> gauging_changes['Links_deleted'].shape
            (17, 11)
            >>> gauging_changes['Nodes_added'].shape
            (2, 4)
            >>> gauging_changes['Nodes_deleted'].shape
            (13, 4)
        """

        path_to_pickle = cd(self.SHP_DIR_PATH, trk_date, self.NM_CHANGES_FILENAME + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            gauging_changes = load_data(path_to_pickle)

        else:
            path_to_file = path_to_pickle.replace(".pickle", ".xlsx")

            if verbose:
                print("Parsing \"{}\"".format(self.NM_CHANGES_DESC), end=" ... ")

            try:
                with pd.ExcelFile(path_to_file) as workbook:
                    sheet_data = [workbook.parse(sheet_name=sheet_name,
                                                 parse_dates=['VERSION_DATE', 'LAST_EDITED_DATE'])
                                  for sheet_name in workbook.sheet_names]

                    for dat in sheet_data:
                        if 'VERSION_DATE' in dat.columns:
                            dat.VERSION_DATE = dat.VERSION_DATE.map(
                                lambda x: datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S.%f'))

                    gauging_changes = dict(zip(workbook.sheet_names, sheet_data))

                    workbook.close()

                if verbose:
                    print("Done.")

                save_data(gauging_changes, path_to_pickle, verbose=verbose)

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")
                gauging_changes = None

        return gauging_changes

    def import_gauging_changes(self, trk_date, update=False, confirmation_required=True,
                               verbose=True, **kwargs):
        # noinspection PyShadowingNames
        """
        Import the shapefile (calibrated) data of UK tracks into the project database.

        :param trk_date: Data date of track shapefiles.
        :type trk_date: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param confirmation_required: Whether to ask for confirmation to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.import_data`_.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> gauging_changes = trk.read_gauging_changes(trk_date='202004')
            >>> type(gauging_changes)
            dict
            >>> list(gauging_changes.keys())
            ['Links_added', 'Links_deleted', 'Nodes_added', 'Nodes_deleted']
            >>> trk.import_gauging_changes(trk_date='202004', if_exists='replace')
            To import the Network Model gauging changes (39pt1 to 39pt2) into the schema "Track"?
             [No]|Yes: yes
            Importing ...
                "Links_added" ... Done.
                "Links_deleted" ... Done.
                "Nodes_added" ... Done.
                "Nodes_deleted" ... Done.

        .. figure:: ../_images/trk_track_links_added_tbl.*
            :name: trk_track_links_added_tbl
            :align: center
            :width: 100%

            Snapshot of the "Track"."Links_added" table.

        .. figure:: ../_images/trk_track_links_deleted_tbl.*
            :name: trk_track_links_deleted_tbl
            :align: center
            :width: 100%

            Snapshot of the "Track"."Links_deleted" table.

        .. figure:: ../_images/trk_track_nodes_added_tbl.*
            :name: trk_track_nodes_added_tbl
            :align: center
            :width: 100%

            Snapshot of the "Track"."Nodes_added" table.

        .. figure:: ../_images/trk_track_nodes_deleted_tbl.*
            :name: trk_track_nodes_deleted_tbl
            :align: center
            :width: 100%

            Snapshot of the "Track"."Nodes_deleted" table.
        """

        dat_name = f"data of {self.NM_CHANGES_DESC}"

        if confirmed(f"To import {dat_name} into the schema \"{self.SCHEMA_NAME}\"?\n",
                     confirmation_required=confirmation_required):

            gauging_changes = self.read_gauging_changes(
                trk_date=trk_date, update=update, verbose=verbose)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing ... ")

            for tbl_name, dat in gauging_changes.items():
                if verbose:
                    print(f"\t\"{tbl_name}\"", end=" ... ")

                try:
                    self.db_instance.import_data(
                        data=dat, schema_name=self.SCHEMA_NAME, table_name=tbl_name,
                        method=self.db_instance.psql_insert_copy, confirmation_required=False,
                        **kwargs)

                    if verbose:
                        print("Done.")

                except Exception as e:
                    _print_failure_msg(e, msg="Failed.")

    def read_ref_line_shp(self, trk_date, update=False, verbose=False, **kwargs):
        # noinspection PyShadowingNames
        """
        Read the shapefile (calibrated) of the reference line from a local directory.

        :param trk_date: Data date of track shapefiles.
        :type trk_date: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the method
            `pydriosm.reader.SHPReadParse.read_shp`_.
        :return: Tabular data of the shapefile (calibrated) of the reference line.
        :rtype: geopandas.GeoDataFrame | pandas.DataFrame

        .. _`pydriosm.reader.SHPReadParse.read_shp`:
            https://pydriosm.readthedocs.io/en/latest/_generated/
            pydriosm.reader.SHPReadParse.read_shp.html

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> ref_line = trk.read_ref_line_shp(trk_date='202004')
            >>> # ref_line = trk.read_ref_line_shp(trk_date='202004', update=True, verbose=True)
            >>> type(ref_line)
            geopandas.geodataframe.GeoDataFrame
            >>> ref_line.shape
            (1583, 22)
        """

        path_to_pickle = cd(self.SHP_DIR_PATH, trk_date, self.REF_LINE_FILENAME + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            ref_line_shp = load_data(path_to_pickle)

        else:
            path_to_shp = path_to_pickle.replace(".pickle", ".shp")

            if verbose:
                print(f"Parsing the {self.REF_LINE_DESC[0].lower() + self.REF_LINE_DESC[1:]}",
                      end=" ... ")

            try:
                kwargs.update({'shp_pathname': path_to_shp, 'engine': 'gpd'})
                ref_line_shp = SHPReadParse.read_shp(**kwargs)

                ref_line_shp['ASSETID'] = ref_line_shp['ASSETID'].astype(np.int64)

                ref_line_shp.rename(columns={'VERSION_NU': 'VERSION_NUMBER',
                                             'VERSION_DA': 'VERSION_DATE',
                                             'EDIT_STATU': 'EDIT_STATUS',
                                             'LAST_EDITE': 'LAST_EDITED_BY',
                                             'LAST_EDI_1': 'LAST_EDITED_DATE',
                                             'CHECKED_DA': 'CHECKED_DATE',
                                             'VALIDATED_': 'VALIDATED_BY',
                                             'VALIDATED1': 'VALIDATED_DATE'}, inplace=True)

                ref_line_shp['VERSION_DATE'] = ref_line_shp['VERSION_DATE'].map(
                    lambda x: datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S.%f'))

                date_cols = ['LAST_EDITED_DATE', 'CHECKED_DATE', 'VALIDATED_DATE']
                ref_line_shp[date_cols] = ref_line_shp[date_cols].map(
                    lambda x: pd.NaT if pd.isna(x)
                    else datetime.datetime.strptime(x, '%Y-%m-%d').date())

                if verbose:
                    print("Done.")

                save_data(ref_line_shp, path_to_file=path_to_pickle, verbose=verbose)

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")
                ref_line_shp = None

        return ref_line_shp

    def import_ref_line_shp(self, trk_date, update=False, confirmation_required=True, verbose=True,
                            **kwargs):
        # noinspection PyShadowingNames
        """
        Import the shapefile (calibrated) of the reference line into the project database.

        :param trk_date: Data date of track shapefiles.
        :type trk_date: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param confirmation_required: Whether to ask for confirmation to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.import_data`_.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> trk.import_ref_line_shp(trk_date='202004')
            To import ref line data into the table "Track"."RefLine_shapefile_calibrated"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/trk_ref_line_shp_tbl.*
            :name: trk_ref_line_shp_tbl
            :align: center
            :width: 100%

            Snapshot of the "Track"."RefLine_shapefile_calibrated" table.
        """

        dat_name = self.REF_LINE_DESC[0].lower() + self.REF_LINE_DESC[1:]
        tbl_name = f"\"{self.SCHEMA_NAME}\".\"{self.REF_LINE_TABLE_NAME}\""

        if confirmed(f"To import the {dat_name} into the table {tbl_name}?\n",
                     confirmation_required=confirmation_required):

            ref_line_shp = self.read_ref_line_shp(trk_date=trk_date, update=update, verbose=verbose)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing the data", end=" ... ")

            try:
                if 'geometry' in ref_line_shp.columns:
                    ref_line_shp = pd.DataFrame(ref_line_shp, copy=True)
                    ref_line_shp['geometry'] = ref_line_shp['geometry'].map(lambda x: x.wkt)

                self.db_instance.import_data(
                    data=ref_line_shp, schema_name=self.SCHEMA_NAME,
                    table_name=self.REF_LINE_TABLE_NAME, method=self.db_instance.psql_insert_copy,
                    confirmation_required=False, **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")

    def read_tracks_shp(self, trk_date, update=False, verbose=False, **kwargs):
        # noinspection PyShadowingNames
        """
        Read the shapefile (calibrated) data of tracks of the whole UK from a local directory.

        :param trk_date: Data date of track shapefiles.
        :type trk_date: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the method
            `pydriosm.reader.SHPReadParse.read_shp`_.
        :return: Tabular data of the shapefile (calibrated) of tracks of the entire UK.
        :rtype: geopandas.GeoDataFrame | pandas.DataFrame | None

        .. _`pydriosm.reader.SHPReadParse.read_shp`:
            https://pydriosm.readthedocs.io/en/latest/_generated/
            pydriosm.reader.SHPReadParse.read_shp.html

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> # trk_shp = trk.read_tracks_shp(trk_date='202004', update=True, verbose=True)
            >>> trk_shp = trk.read_tracks_shp(trk_date='202004')
            >>> type(trk_shp)
            geopandas.geodataframe.GeoDataFrame
            >>> trk_shp.shape
            (49667, 11)
            >>> # example_record = trk_shp.loc[[10545], :]
            >>> # example_record
        """

        path_to_pickle = cd(self.SHP_DIR_PATH, trk_date, self.TRK_SHP_FILENAME + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            tracks_shp = load_data(path_to_pickle)

        else:
            path_to_shp = path_to_pickle.replace(".pickle", ".shp")

            if verbose:
                print(f"Parsing the {self.TRK_SHP_DESC[0].lower() + self.TRK_SHP_DESC[1:]}",
                      end=" ... ")

            try:
                kwargs.update({'shp_pathname': path_to_shp, 'engine': 'gpd'})
                tracks_shp = SHPReadParse.read_shp(**kwargs)

                int_cols = ['ASSETID', 'L_LINK_ID', 'TRID', 'TRCODE']
                tracks_shp[int_cols] = tracks_shp[int_cols].astype(np.int64)

                if verbose:
                    print("Done.")

                save_data(tracks_shp, path_to_file=path_to_pickle, verbose=verbose)

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")
                tracks_shp = None

        return tracks_shp

    def import_tracks_shp(self, trk_date, update=False, confirmation_required=True, verbose=True,
                          **kwargs):
        # noinspection PyShadowingNames
        """
        Import the shapefile (calibrated) data of UK tracks into the project database.

        :param trk_date: Data date of track shapefiles.
        :type trk_date: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param confirmation_required: Whether to ask for confirmation to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.import_data`_.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> trk.import_tracks_shp(trk_date='202004')
            To import track shapefile into the table "Track"."Tracks_shapefile_calibrated"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/trk_track_shp_tbl.*
            :name: trk_track_shp_tbl
            :align: center
            :width: 100%

            Snapshot of the "Track"."Tracks_shapefile_calibrated" table.
        """

        dat_name = self.TRK_SHP_DESC[0].lower() + self.TRK_SHP_DESC[1:]
        tbl_name = f"\"{self.SCHEMA_NAME}\".\"{self.TRK_SHP_TABLE_NAME}\""

        if confirmed(f"To import {dat_name} into the table {tbl_name}?\n", confirmation_required):
            tracks_shp = self.read_tracks_shp(trk_date=trk_date, update=update, verbose=verbose)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing the data", end=" ... ")

            try:
                if 'geometry' in tracks_shp.columns:
                    tracks_shp['geometry'] = tracks_shp['geometry'].map(lambda x: x.wkt)

                tracks_shp.insert(0, 'Year', int(trk_date[:4]))
                tracks_shp.insert(1, 'Month', int(trk_date[-2:]))

                self.db_instance.import_data(
                    data=tracks_shp, schema_name=self.SCHEMA_NAME,
                    table_name=self.TRK_SHP_TABLE_NAME, method=self.db_instance.psql_insert_copy,
                    confirmation_required=False, **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")

    def fetch_tracks_shapefiles(self, trk_date, update=False, verbose=False, **kwargs):
        # noinspection PyShadowingNames
        """
        Fetch data of track shapefiles (calibrated) from a local directory.

        :param trk_date: Data date of track shapefiles.
        :type trk_date: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the method
            `pydriosm.reader.SHPReadParse.read_shp`_.
        :return: Data of the track shapefiles.
        :rtype: dict | None

        .. _`pydriosm.reader.SHPReadParse.read_shp`:
            https://pydriosm.readthedocs.io/en/latest/_generated/
            pydriosm.reader.SHPReadParse.read_shp.html

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> # trk_shp_data = trk.fetch_tracks_shapefiles('202004', update=True, verbose=True)
            >>> trk_shp_data = trk.fetch_tracks_shapefiles(trk_date='202004')
            >>> type(trk_shp_data)
            dict
            >>> list(trk_shp_data.keys())
            ['UK Tracks', 'Network Model Changes', 'Reference Line']
            >>> trk_shp_data['UK Tracks'].shape
            (49667, 11)
            >>> type(trk_shp_data['Network Model Changes'])
            dict
            >>> list(trk_shp_data['Network Model Changes'].keys())
            ['Links_added', 'Links_deleted', 'Nodes_added', 'Nodes_deleted']
            >>> trk_shp_data['Network Model Changes']['Links_added'].shape
            (3, 11)
            >>> trk_shp_data['Network Model Changes']['Links_deleted'].shape
            (17, 11)
            >>> trk_shp_data['Network Model Changes']['Nodes_added'].shape
            (2, 4)
            >>> trk_shp_data['Network Model Changes']['Nodes_deleted'].shape
            (13, 4)
            >>> trk_shp_data['Reference Line'].shape
            (1583, 22)
        """

        pickle_filename = "Track_shape_data.pickle"
        path_to_pickle = cd(self.SHP_DIR_PATH, trk_date, pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            track_shp_data = load_data(path_to_pickle)

        else:
            if verbose:
                print("Fetching data of track shapefiles (calibrated) ... ")

            try:
                track_shp_data = {
                    trk_date: {
                        'UK Tracks':
                            self.read_tracks_shp(trk_date, update, verbose, **kwargs),
                        'Network Model Changes':
                            self.read_gauging_changes(trk_date, update, verbose, **kwargs),
                        'Reference Line':
                            self.read_ref_line_shp(trk_date, update, verbose, **kwargs)
                    }
                }

                save_data(track_shp_data, path_to_pickle, verbose=verbose)

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")
                track_shp_data = None

        return track_shp_data

    def import_tracks_shapefiles(self, trk_date, update=False, confirmation_required=True,
                                 verbose=True, **kwargs):
        # noinspection PyShadowingNames
        """
        Import data of track shapefiles into the project database.

        :param trk_date: Data date of track shapefiles.
        :type trk_date: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param confirmation_required: Whether to ask for confirmation to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.import_data`_.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> trk.import_tracks_shapefiles(trk_date='202004', if_exists='replace')
            To import data of track shapefiles into the schema "Track"?
             [No]|Yes: yes
            Importing ...
                Shapefile (calibrated) of tracks of the whole UK ... Done.
                Network Model gauging changes (39pt1 to 39pt2) ... Done.
                Shapefile (calibrated) of reference line ... Done.

        .. seealso::

            - Examples for the mothds :meth:`~src.preprocessor.track.Track.import_gauging_changes`,
              :meth:`~src.preprocessor.track.Track.import_ref_line_shp` and
              :meth:`~src.preprocessor.track.Track.import_tracks_shp`.
        """

        if confirmed(
                f"To import data of track shapefiles into the schema \"{self.SCHEMA_NAME}\"?\n",
                confirmation_required=confirmation_required):

            track_shp_data = {
                'UK Tracks': self.read_tracks_shp(
                    trk_date=trk_date, update=update, verbose=verbose),
                'Network Model Changes':
                    self.read_gauging_changes(trk_date=trk_date, update=update, verbose=verbose),
                'Reference Line':
                    self.read_ref_line_shp(trk_date=trk_date, update=update, verbose=verbose)
            }

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            print("Importing ... ") if verbose else ""

            for dat_name, dat in track_shp_data.items():

                if 'geometry' in dat.columns:
                    # dat = pd.DataFrame(dat)
                    dat['geometry'] = dat['geometry'].map(lambda x: x.wkt)

                try:
                    if dat_name == 'UK Tracks':
                        if verbose:
                            print(f"\t{self.TRK_SHP_DESC}", end=" ... ")

                        self.db_instance.import_data(
                            data=dat, schema_name=self.SCHEMA_NAME,
                            table_name=self.TRK_SHP_TABLE_NAME,
                            method=self.db_instance.psql_insert_copy, confirmation_required=False,
                            **kwargs)

                    elif dat_name == 'Network Model Changes':
                        if verbose:
                            print(f"\t{self.NM_CHANGES_DESC}", end=" ... ")

                        for tbl_, dat_ in dat.items():
                            self.db_instance.import_data(
                                data=dat_, schema_name=self.SCHEMA_NAME, table_name=tbl_,
                                method=self.db_instance.psql_insert_copy,
                                confirmation_required=False, **kwargs)

                    else:
                        if verbose:
                            print(f"\t{self.REF_LINE_DESC}", end=" ... ")

                        self.db_instance.import_data(
                            data=dat, schema_name=self.SCHEMA_NAME,
                            table_name=self.REF_LINE_TABLE_NAME,
                            method=self.db_instance.psql_insert_copy, confirmation_required=False,
                            **kwargs)

                    if verbose:
                        print("Done.")

                except Exception as e:
                    _print_failure_msg(e, msg="Failed.")

    def load_tracks_shp(self, trk_date=None, elr=None, **kwargs):
        # noinspection PyShadowingNames
        """
        Load the shapefile (calibrated) data of UK tracks from the project database.

        :param trk_date: Data date of track shapefiles.
        :type trk_date: str | int | None
        :param elr: Engineer's Line Reference(s); defaults to ``None``.
        :type elr: str | list | tuple | None
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.read_sql_query`_.
        :return: Shapefile (calibrated) data of UK tracks.
        :rtype: pandas.DataFrame

        .. _`pyhelpers.dbms.PostgreSQL.read_sql_query`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.read_sql_query.html

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> trk_shp = trk.load_tracks_shp(trk_date='202004', elr=['ECM7', 'ECM8'])
            >>> trk_shp.shape
            (440, 13)
        """

        if self.db_instance is None:
            self.db_instance = TrackFixityDB()

        sql_query = f'SELECT * FROM "{self.SCHEMA_NAME}"."{self.TRK_SHP_TABLE_NAME}"'

        sql_query = add_sql_query_date_condition(sql_query, data_date=trk_date)

        sql_query = add_sql_query_elr_condition(sql_query, elr=elr)

        sql_query += 'ORDER BY "Year" ASC, "Month" ASC, "ASSETID" ASC'

        tracks_shp = self.db_instance.read_sql_query(sql_query=sql_query, **kwargs)

        tracks_shp.geometry = tracks_shp.geometry.map(shapely.wkt.loads)

        return tracks_shp

    def make_pseudo_mileage_dict(self, elr=None, set_index=True):
        # noinspection PyShadowingNames
        """
        Make a dictionary of (pseudo) mileages for all available track IDs
        based on track shapefiles.

        :param elr: (One) ELR; defaults to ``None``.
        :type elr: str | list | None
        :param set_index: Whether to set ``'pseudo_geometry'`` to be the index for the dataframe
            of pseudo track mileage; defaults to ``True``.
        :type set_index: bool
        :return: Mileages for all available track IDs from track shapefiles.
        :rtype: dict

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> pseudo_track_mileage_dict = trk.make_pseudo_mileage_dict(elr='ECM8')
            >>> list(pseudo_track_mileage_dict.keys())
            ['ECM8']
            >>> list(pseudo_track_mileage_dict['ECM8'].keys())[:5]
            [1100, 1200, 1700, 1900, 1901]
            >>> pseudo_track_mileage_dict['ECM8'][1100].shape
            (94899, 1)
        """

        tracks_shp_data = self.load_tracks_shp(elr=elr)

        def _make_pseudo_trk_mileage(l_m_from, l_m_to, geometry):
            # l_m_from, l_m_to, geometry = dat.L_M_FROM[1], dat.L_M_TO[1], dat.geometry[1]
            s_yard, e_yard = map(mileage_to_yard, (l_m_from, l_m_to))

            pseudo_mil_ = map(yard_to_mileage, np.arange(s_yard, e_yard + 1))
            pseudo_mil = list(map(mileage_str_to_num, pseudo_mil_))

            geometry_ = shapely.ops.transform(lambda x, y: (x, y, 0), geometry)
            pseudo_mil_geom = [
                geometry_.interpolate((j / (e_yard - s_yard)), normalized=True).coords[0]
                for j in range(1, e_yard - s_yard)]

            a, b = geometry_.boundary.geoms
            pseudo_mil_geom = [a.coords[0]] + pseudo_mil_geom + [b.coords[0]]

            # fig = plt.figure()
            # ax = fig.add_subplot()
            # geom_arr = np.array(geometry_)
            # ax.scatter(geom_arr[:, 0], geom_arr[:, 1], color='#ff7f0e', label='track shapefile')
            # pseudo_mil_geom_arr = np.array(LineString(pseudo_mil_geom))
            # ax.plot(pseudo_mil_geom_arr[:, 0], pseudo_mil_geom_arr[:, 1], label='pseudo')
            # ax.legend()
            # plt.tight_layout()

            pseudo_dat = {'pseudo_mileage': pseudo_mil, 'pseudo_geometry': pseudo_mil_geom}
            pseudo_data = pd.DataFrame(pseudo_dat)
            pseudo_data['pseudo_geometry'] = pseudo_data['pseudo_geometry'].map(
                shapely.geometry.Point)

            return pseudo_data

        tracks_shp_dict = dict(list(tracks_shp_data.groupby('ELR')))

        pseudo_trk_shp_mileage_dict = tracks_shp_dict.copy()

        for trk_elr, tracks_shp in tracks_shp_dict.items():

            pseudo_trk_shp_mil_dict = {}
            for tid, dat in tracks_shp.groupby('TRID'):
                # Test:
                # mileage_columns = ['L_M_FROM', 'L_M_TO']
                # dat_ = pd.DataFrame(dat.geometry.map(lambda x: x.coords.xy).values.tolist())
                # trk_shp = dat_.map(shapely.geometry.Point)
                # trk_shp.columns = [col + '_geom' for col in mileage_columns]
                # trk_shp = dat.join(trk_shp)

                pseudo_trk_mileage_list = []
                for i in dat.index:
                    m_from, m_to, geom = dat.loc[i, ['L_M_FROM', 'L_M_TO', 'geometry']].values
                    pseudo_trk_mileage_list.append(
                        _make_pseudo_trk_mileage(l_m_from=m_from, l_m_to=m_to, geometry=geom))

                pseudo_track_mileage = pd.concat(pseudo_trk_mileage_list)

                pseudo_track_mileage.drop_duplicates(subset=['pseudo_mileage'], inplace=True)

                # fig = plt.figure()
                # ax = fig.add_subplot()
                # geom_arr = np.array(LineString(pseudo_track_mileage.pseudo_geometry.to_list()))
                # ax.plot(geom_arr[:, 0], geom_arr[:, 1], label='pseudo track shapefile')

                if set_index:
                    pseudo_track_mileage.set_index('pseudo_mileage', inplace=True)

                pseudo_trk_shp_mil_dict.update({tid: pseudo_track_mileage})

            # noinspection PyTypeChecker
            pseudo_trk_shp_mileage_dict[trk_elr] = pseudo_trk_shp_mil_dict

            # fig = plt.figure()
            # ax = fig.add_subplot()
            # for geom_data in pseudo_trk_shp_mileage_dict.values():
            #     for geom_dat in geom_data.values():
            #         for geom_ in geom_dat.pseudo_geometry:
            #             ax.scatter(geom_.x, geom_.y, s=2)

        return pseudo_trk_shp_mileage_dict

    def import_pseudo_mileage_dict(self, elr=None, set_index=True, confirmation_required=True,
                                   verbose=True, **kwargs):
        # noinspection PyShadowingNames
        """
        Import data of (pseudo) mileages into the project database.

        :param elr: (One) ELR; defaults to ``None``.
        :type elr: str | list | None
        :param set_index: Whether to set ``'pseudo_geometry'`` to be the index;
            defaults to ``True``.
        :type set_index: bool
        :param confirmation_required: Whether asking for confirmation to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the
            `pyhelpers.dbms.PostgreSQL.import_data`_ method.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> trk.import_pseudo_mileage_dict(elr=['ECM7', 'ECM8'])
            To import pseudo mileages (ECM7, ECM8) into the schema "Track"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/trk_pseudo_mileages_tbl.*
            :name: trk_pseudo_mileages_tbl
            :align: center
            :width: 100%

            Snapshot of the "Track"."Pseudo_mileage" table.
        """

        pseudo_mileage_dict = self.make_pseudo_mileage_dict(elr=elr, set_index=set_index)

        dat_name = 'pseudo mileages ({})'.format(', '.join(list(pseudo_mileage_dict.keys())))
        if confirmed(f"To import {dat_name} into the schema \"{self.SCHEMA_NAME}\"?\n",
                     confirmation_required=confirmation_required):

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing the data", end=" ... ")

            try:
                data_list = []
                for elr_, elr_dat in pseudo_mileage_dict.items():
                    dat_list = []
                    for tid, dat_ in elr_dat.items():
                        dat_['TRID'] = tid
                        dat_list.append(dat_)
                    data = pd.concat(dat_list)
                    data['ELR'] = elr_
                    data_list.append(data)

                data = pd.concat(data_list)

                self.db_instance.import_data(
                    data=data, schema_name=self.SCHEMA_NAME,
                    table_name=self.PSEUDO_MILEAGE_TABLE_NAME,
                    method=self.db_instance.psql_insert_copy, confirmation_required=False,
                    index=True, **kwargs)

                print("Done.") if verbose else ""

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")

    def load_pseudo_mileage_dict(self, elr=None, **kwargs):
        # noinspection PyShadowingNames
        """
        Load the data of (pseudo) mileages from the project database.

        :param elr: Engineer's Line Reference(s); defaults to ``None``.
        :type elr: str | list | tuple | None
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.read_sql_query`_.
        :return: Data of pseudo mileages (based on track shapefile).
        :rtype: dict

        .. _`pyhelpers.dbms.PostgreSQL.read_sql_query`:
            https://pyhelpers.readthedocs.io/en/latest/
            _generated/pyhelpers.dbms.PostgreSQL.read_sql_query.html

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> pseudo_track_mileage_dict = trk.load_pseudo_mileage_dict(elr=['ECM7', 'ECM8'])
            >>> type(pseudo_track_mileage_dict)
            dict
            >>> list(pseudo_track_mileage_dict.keys())
            ['ECM7', 'ECM8']
            >>> list(pseudo_track_mileage_dict['ECM8'].keys())[:5]
            [1100, 1200, 1700, 1900, 1901]
            >>> pseudo_track_mileage_dict['ECM8'][1100].shape
            (94899, 1)
        """

        if self.db_instance is None:
            self.db_instance = TrackFixityDB()

        sql_query = f'SELECT * FROM "{self.SCHEMA_NAME}"."{self.PSEUDO_MILEAGE_TABLE_NAME}"'

        sql_query = add_sql_query_elr_condition(sql_query, elr=elr)

        pseudo_mileage_data = self.db_instance.read_sql_query(sql_query=sql_query, **kwargs)

        pseudo_mileage_dict_ = dict(list(pseudo_mileage_data.groupby('ELR')))

        pseudo_mileage_dict = copy.copy(pseudo_mileage_dict_)
        for elr_, dat in pseudo_mileage_dict_.items():
            dat_ = dat.drop(columns=['ELR'])

            temp = []
            for tid, tid_dat in dat_.groupby('TRID'):
                tid_dat.pseudo_geometry = tid_dat.pseudo_geometry.map(shapely.wkt.loads)
                tid_dat_ = tid_dat.drop(columns=['TRID']).set_index('pseudo_mileage')
                temp.append((tid, tid_dat_))

            # noinspection PyTypeChecker
            pseudo_mileage_dict[elr_] = dict(temp)

        return pseudo_mileage_dict

    # == Track quality files =======================================================================

    @staticmethod
    def fmt_dtf_file_date(tq_date):
        # noinspection PyShadowingNames
        """
        Reformat the date (in the filename) of a track quality file.

        :param tq_date: Date (in the filename) of a track quality file.
        :type tq_date: str
        :return: Reformatted date of the track quality file.
        :rtype: datetime.date | datetime.datetime

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> trk.fmt_dtf_file_date(tq_date='02-12-2019')
            datetime.date(2019, 12, 2)
        """

        tq_date_ = datetime.datetime.strptime(tq_date, '%d-%m-%Y').date()

        return tq_date_

    def get_dtf_pathname(self, tid, tq_date):
        # noinspection PyShadowingNames
        """
        Get the path to a track quality file of a given track ID and date.

        :param tid: Track ID.
        :type tid: str
        :param tq_date: Date of a track quality file.
        :type tq_date: str | datetime.date
        :return: Pathname of the track quality file of the given ``tid`` and ``tq_date``.
        :rtype: str

        **Examples**::

            >>> from src.preprocessor import Track
            >>> import os
            >>> trk = Track()
            >>> dtf_file_path = trk.get_dtf_pathname(tid='1100', tq_date='02-12-2019')
            >>> os.path.isfile(dtf_file_path)
            True
        """

        if isinstance(tq_date, str):
            tq_date = self.fmt_dtf_file_date(tq_date)
        data_dates = [self.fmt_dtf_file_date(x) for x in self.dtf_data_dates[tid]]

        tqd = min(data_dates, key=lambda x: abs(x - tq_date))
        tq_date = datetime.datetime.strftime(tqd, '%d-%m-%Y')

        file_path = cd(self.DTF_DIR_PATH, tid, f"{tid}_{tq_date}.dtf")

        return file_path

    def parse_dtf(self, path_to_dtf):
        # noinspection PyShadowingNames
        """
        Parse a track quality file (.dtf format).

        :param path_to_dtf: Path to the track quality .dtf file.
        :type path_to_dtf: str
        :return: Data parsed from the track quality .dtf file.
        :rtype: dict

        **Examples**::

            >>> from src.preprocessor import Track
            >>> import os
            >>> trk = Track()
            >>> path_to_dtf_file = trk.dtf_pathnames[0]
            >>> os.path.isfile(path_to_dtf_file)
            True
            >>> dtf_dat = trk.parse_dtf(path_to_dtf_file)
            >>> type(dtf_dat)
            dict
            >>> list(dtf_dat.keys())
            ['TQ', 'ELR', 'Track Id', 'Start Eighth', 'End Eighth', 'Date(DD/MM/YYYY)']
            >>> dtf_dat['TQ'].shape
            (334844, 18)
        """

        with open(path_to_dtf) as f:
            counter = 0
            info = []
            while counter <= 3:
                line = f.readline().strip('\n')
                if line:
                    info.append(line.split('\t'))
                counter += 1

        desc, cols = info[0][0].split('    '), info[1:]

        col_names = ['{} ({})'.format(x, y) for x, y in zip(*cols)]
        dtf = pd.read_csv(path_to_dtf, sep='\t', skiprows=4, header=None, names=col_names)

        dtf_data = {self.DTF_KEY: dtf}

        # noinspection PyTypeChecker
        dtf_data.update(dict([x.split(': ') for x in desc]))

        return dtf_data

    def read_dtf(self, tid, tq_date, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read track quality files (.dtf) for a given pair of track ID and date
        from a local directory.

        :param tid: Track ID.
        :type tid: int | str
        :param tq_date: Date of the track quality file.
        :type tq_date: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of the track quality file, or ``None`` if the file is not found.
        :rtype: dict | None

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> trk.track_quality_tid
            ['1100', '2100']
            >>> trk.dtf_data_dates
            {'1100': ['14-10-2019', '02-12-2019', '27-04-2020'],
             '2100': ['14-10-2019', '06-01-2020', '30-03-2020']}
            >>> # tqf_data = trk.read_dtf('1100', '14-10-2019', update=True, verbose=True)
            >>> tqf_data = trk.read_dtf(tid='1100', tq_date='14-10-2019')
            >>> type(tqf_data)
            dict
            >>> list(tqf_data.keys())
            ['TQ', 'ELR', 'Track Id', 'Start Eighth', 'End Eighth', 'Date(DD/MM/YYYY)']
            >>> tqf_data['TQ'].shape
            (334844, 18)
        """

        path_to_dtf = self.get_dtf_pathname(tid, tq_date)

        if not os.path.isfile(path_to_dtf):
            print("The specified file does not exist.")

        else:
            path_to_pickle = path_to_dtf.replace(".dtf", ".pickle")

            if os.path.isfile(path_to_pickle) and not update:
                dtf_data = load_data(path_to_pickle)

            else:
                if verbose:
                    print(f"Parsing the track quality file \"{os.path.basename(path_to_dtf)}\"",
                          end=" ... ")

                try:
                    dtf_data = self.parse_dtf(path_to_dtf)

                    if verbose:
                        print("Done.")

                    save_data(dtf_data, path_to_pickle, verbose=verbose)

                except Exception as e:
                    _print_failure_msg(e, msg="Failed.")
                    dtf_data = None

            return dtf_data

    def import_dtf(self, tid, tq_date, update=False, confirmation_required=True, verbose=True,
                   **kwargs):
        # noinspection PyShadowingNames
        """
        Import data of a track quality file (of a given track ID and date)
        into the project database.

        :param tid: Track ID.
        :type tid: int | str
        :param tq_date: Date of the track quality file.
        :type tq_date: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param confirmation_required: Whether asking for confirmation to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.import_data`_.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> trk.import_dtf(tid='2100', tq_date='30-03-2020')
            To import "2100_30-03-2020.dtf" into the table "Track"."Track_quality_2100_30-03-2020"?
             [No]|Yes: yes
            Importing the data into  ... Done.
        """

        filename = os.path.basename(self.get_dtf_pathname(tid=tid, tq_date=tq_date))

        table_name = self.DTF_TABLE_NAME + '_' + filename.replace('.dtf', '')

        tbl_name = f'"{self.SCHEMA_NAME}"."{table_name}"'

        if confirmed(f"To import \"{filename}\" into the table {tbl_name}?\n",
                     confirmation_required=confirmation_required):
            dat = self.read_dtf(tid=tid, tq_date=tq_date, update=update, verbose=verbose)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing the data", end=" ... ")

            try:
                self.db_instance.import_data(
                    data=dat[self.DTF_KEY], schema_name=self.SCHEMA_NAME, table_name=table_name,
                    method=self.db_instance.psql_insert_copy, confirmation_required=False,
                    **kwargs)

                dat.pop(self.DTF_KEY)
                metadata = pd.DataFrame(dat.items(), columns=['Attribute', 'Value'])

                self.db_instance.import_data(
                    data=metadata, schema_name=self.SCHEMA_NAME,
                    table_name=table_name + '_Metadata', confirmation_required=False,
                    **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")

    def read_track_quality(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read data of all available track quality files from a local directory.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of all available track quality files.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> # tqf_data = trk.read_track_quality(update=True, verbose=True)
            >>> tqf_data = trk.read_track_quality()
            >>> tqf_data.shape
            (2093003, 23)
        """

        path_to_pickle = cd(self.DTF_DIR_PATH, "track_quality_files.pickle")

        if os.path.isfile(path_to_pickle) and not update:
            tq_files_data = load_data(path_to_pickle)

        else:
            tq_files_dat = []
            for tid, tq_dates in self.dtf_data_dates.items():
                for tq_date in tq_dates:
                    tq_file_data = self.read_dtf(
                        tid=tid, tq_date=tq_date, update=update, verbose=verbose)

                    tq_file_dat = tq_file_data[self.DTF_KEY].copy()
                    i = 0
                    for k, v in tq_file_data.items():
                        if k != self.DTF_KEY:
                            if k.startswith('Date'):
                                k = 'Date'
                                v = datetime.datetime.strptime(v, '%d/%m/%Y').date()
                            tq_file_dat.insert(i, k, v)
                            i += 1

                    tq_files_dat.append(tq_file_dat)

            tq_files_data = pd.concat(tq_files_dat, axis=0, ignore_index=True)
            tq_files_data.sort_values(by=['Track Id', 'Date'], inplace=True)
            tq_files_data.index = range(len(tq_files_data))

            save_data(pickle_data=tq_files_data, path_to_file=path_to_pickle, verbose=verbose)

        return tq_files_data

    def import_track_quality(self, update=False, confirmation_required=True, verbose=True,
                             **kwargs):
        # noinspection PyShadowingNames
        """
        Import data of all track quality files into the project database.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param confirmation_required: Whether asking for confirmation to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.import_data`_.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> trk.import_track_quality()
            To import track quality files into the table "Track"."Track_quality"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/trk_track_quality_tbl.*
            :name: trk_track_quality_tbl
            :align: center
            :width: 100%

            Snapshot of the "Track"."Track_quality" table.
        """

        tbl_name = f'"{self.SCHEMA_NAME}"."{self.DTF_TABLE_NAME}"'

        if confirmed(f"To import track quality files into the table {tbl_name}?\n",
                     confirmation_required=confirmation_required):

            tq_files_data = self.read_track_quality(update=update, verbose=verbose)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing the data", end=" ... ")

            try:
                self.db_instance.import_data(
                    data=tq_files_data, schema_name=self.SCHEMA_NAME,
                    table_name=self.DTF_TABLE_NAME, method=self.db_instance.psql_insert_copy,
                    confirmation_required=False, **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")

    def load_track_quality(self, elr=None, tq_date=None, **kwargs):
        # noinspection PyShadowingNames
        """
        Load the data of track quality files from the project database.

        :param elr: Engineer's Line Reference(s); defaults to ``None``.
        :type elr: str | list | tuple | None
        :param tq_date: Date of a track quality file, formatted as 'YYYY-MM-DD';
            defaults to ``None``.
        :type tq_date: str | list | None
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.read_sql_query`_.
        :return: Data of track quality files.
        :rtype: pandas.DataFrame

        .. _`pyhelpers.dbms.PostgreSQL.read_sql_query`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.read_sql_query.html

        **Examples**::

            >>> from src.preprocessor import Track
            >>> trk = Track()
            >>> tqf_data = trk.load_track_quality(elr=['ECM7', 'ECM8'])
            >>> tqf_data.shape
            (1375430, 23)
        """

        if self.db_instance is None:
            self.db_instance = TrackFixityDB()

        sql_query = f'SELECT * FROM "{self.SCHEMA_NAME}"."{self.DTF_TABLE_NAME}"'

        if tq_date is None:
            # tq_dates_ = list(itertools.chain.from_iterable(self.DtfDataDates.values()))
            # tq_dates = [datetime.datetime.strptime(x, '%d-%m-%Y') for x in tq_dates_]
            tq_dates_ = ['2019-10-15', '2020-04-15']
        elif isinstance(tq_date, str):
            tq_dates_ = [tq_date]
        else:
            tq_dates_ = tq_date.copy()
        tq_dates = [
            (f'date(\'{x}\') - INTERVAL \'1 month\'', f'date(\'{x}\') + INTERVAL \'1 month\'')
            for x in tq_dates_]
        sql_query += ' WHERE ("Date" BETWEEN {} AND {}) OR ("Date" BETWEEN {} AND {})'.format(
            *itertools.chain(*tq_dates))

        sql_query = add_sql_query_elr_condition(sql_query, elr=elr)

        sql_query += 'ORDER BY "ELR" ASC, "Locn (mile)" ASC, "Locn (yards)" ASC'

        tq_files_data = self.db_instance.read_sql_query(sql_query=sql_query, **kwargs)

        return tq_files_data


# if __name__ == '__main__':
#     trk = Track()
#
#     _ = trk.read_gauging_changes(trk_date='202004', update=True, verbose=True)
#
#     trk.import_gauging_changes(trk_date='202004', if_exists='replace', verbose=True)
