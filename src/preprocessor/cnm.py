"""
A class for preprocessing data from *Corporate Network Model (CNM)*.
"""

import copy
import glob
import os
import re
import warnings

import pandas as pd
import shapely.wkt
from pyhelpers._cache import _print_failure_msg
from pyhelpers.dirs import cd, cdd
from pyhelpers.ops import confirmed
from pyhelpers.store import load_data, save_data
from pyhelpers.text import extract_words1upper, get_acronym
from pyrcs.converter import mileage_num_to_str
from shapely.geometry import LineString

from src.utils.dgn import (dgn2shp_batch, dgn_shapefiles, dgn_shp_map_view, get_dgn_shp_prj,
                           read_dgn_shapefile)
from src.utils.general import TrackFixityDB, add_sql_query_elr_condition, validate_column_names


def find_item_name(filename):
    item_name_ = re.search(r'(?<=[_.])[A-Z][a-z]+_?([A-Z][a-z]+)?(?=.dgn)', filename)
    if item_name_:
        return item_name_.group(0)


class CNM:
    """
    *Corporate Network Model*.

    The model is Network Rail's geospatial information system.

    .. note::

        This class currently handles only DGN data of waymarks.
    """

    #: Data name.
    NAME: str = 'Corporate Network Model'
    #: Acronym for the data name.
    ACRONYM: str = get_acronym(NAME, only_capitals=True)

    #: Pathname of a local directory where the CNM data is stored.
    DATA_DIR: str = os.path.relpath(cdd(ACRONYM))
    #: Directory name of projection/boundary shapefiles.
    PROJ_DIRNAME: str = "Projection"
    #: Name of the projection file.
    PROJ_FILENAME: str = "Order_69705_Polygon"
    #: Directory name of *waymarks* data.
    WM_DIRNAME: str = "Waymarks"

    #: Name of the schema for storing the CNM data.
    SCHEMA_NAME: str = ACRONYM
    #: Name of the table for storing the *waymarks* data.
    WM_TABLE_NAME: str = WM_DIRNAME.replace(' ', '_')

    def __init__(self, db_instance=None):
        """
        :param db_instance: PostgreSQL database instance; defaults to ``None``.
        :type db_instance: TrackFixityDB | None

        :ivar list wm_dgn_pathnames: List of paths to the original DGN data files.
        :ivar list wm_shp_pathnames: List of lists, each containing paths to shapefiles converted
            from DGN.
        :ivar list wm_item_names: List of names of items available in this data category.
        :ivar TrackFixityDB db_instance: PostgreSQL database instance.


        **Examples**::

            >>> from src.preprocessor import CNM
            >>> cnm = CNM()
            >>> cnm.NAME
            'Corporate Network Model'

        .. figure:: ../_images/cnm_schema.*
            :name: cnm_schema
            :align: center
            :width: 100%

            Snapshot of the *CNM* schema.
        """

        self.wm_dgn_pathnames = glob.glob(cd(self.DATA_DIR, "*", "*.dgn"))

        self.wm_shp_pathnames = [
            [f.replace(".", "_") + "_" + x + ".shp" for x in dgn_shapefiles()]
            for f in self.wm_dgn_pathnames]

        self.wm_item_names = [
            extract_words1upper(find_item_name(f), join_with=' ') for f in self.wm_dgn_pathnames]

        self.db_instance = db_instance

    def read_prj_metadata(self, update=False, parser='osr', as_dict=True, verbose=False):
        """
        Read metadata of projection for the DGN files/shapefiles from a local directory.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param parser: Name of the package used to read the PRJ file;
            options include ``{'osr', 'pycrs'}``; defaults to ``'osr'``.
        :type parser: str
        :param as_dict: Whether to return the data as a dictionary; defaults to ``True``.
        :type as_dict: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Metadata of the projection for the DGN files/shapefiles.
        :rtype: dict | None

        **Examples**::

            >>> from src.preprocessor import CNM
            >>> cnm = CNM()
            >>> dgn_shp_prj = cnm.read_prj_metadata()
            >>> type(dgn_shp_prj)
            dict
            >>> list(dgn_shp_prj.keys())
            ['PROJCS', 'Shapefile']
            >>> list(dgn_shp_prj['PROJCS'].keys())
            ['proj', 'lat_0', 'lon_0', 'k', 'x_0', 'y_0', 'ellps', 'units']
            >>> dgn_shp_prj['Shapefile']
              ORDER_ID  ...                                           geometry
            0    69705  ...  POLYGON ((257789.191 1047540.101, 473308.767 1...
            [1 rows x 4 columns]
        """

        path_to_file = cd(self.DATA_DIR, self.PROJ_DIRNAME, self.PROJ_FILENAME)

        dgn_shp_prj_file = get_dgn_shp_prj(
            path_to_file=path_to_file, update=update, projcs_parser=parser, as_dict=as_dict,
            verbose=verbose)

        return dgn_shp_prj_file

    def dgn2shp(self, dat_name='Waymarks', confirmation_required=True, verbose=True, **kwargs):
        """
        Convert DGN files of CNM data to shapefiles.

        :param dat_name: Name of the data; defaults to ``'Waymarks'``.
        :type dat_name: str | None
        :param confirmation_required: Whether confirmation is required to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the function
            :func:`~src.utils.dgn.dgn2shp`.

        **Examples**::

            >>> from src.preprocessor import CNM
            >>> cnm = CNM()
            >>> cnm.dgn2shp()
            To convert .dgn files of "Waymarks" to shapefiles?
             [No]|Yes: yes
            Converting "CNM_ADMIN.NetworkWaymarks.dgn" at "data\\CNM\\Waymarks" ... Done.
        """

        if dat_name in [self.WM_DIRNAME]:
            dgn2shp_batch(
                dat_name=dat_name, file_paths=glob.glob(cd(self.DATA_DIR, dat_name, "*.dgn")),
                confirmation_required=confirmation_required, verbose=verbose, **kwargs)

        elif dat_name is None:
            if confirmed(
                    f"To convert all available .dgn files of \"{self.ACRONYM}\" to shapefiles?\n",
                    confirmation_required=confirmation_required):
                paths_to_dgn_files = glob.glob(cd(self.DATA_DIR, "*", "*.dgn"))

                dgn2shp_batch(
                    self.ACRONYM, paths_to_dgn_files, confirmation_required=False, verbose=verbose,
                    **kwargs)

    def read_waymarks_shp(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read shapefile data (converted from the DGN data) of waymarks from a local directory.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :return: DGN-converted shapefile data of waymarks.
        :rtype: dict

        **Examples**::

            >>> from src.preprocessor import CNM
            >>> cnm = CNM()
            >>> waymarks_shp = cnm.read_waymarks_shp()
            >>> type(waymarks_shp)
            dict
            >>> list(waymarks_shp.keys())
            ['Annotation', 'MultiPatch', 'Point', 'Polygon', 'Polyline']
            >>> waymarks_shp['Annotation'].empty
            True
            >>> waymarks_shp['MultiPatch'].empty
            True
            >>> waymarks_shp['Point'].empty
            False
            >>> waymarks_shp['Polygon'].empty
            True
            >>> waymarks_shp['Polyline'].empty
            True
            >>> waymarks_shp['Point'].head()
                  Entity  ...                               geometry
            0      Point  ...   POINT Z (150011.002 31282.015 0.000)
            1      Point  ...   POINT Z (150804.322 31339.413 0.000)
            2      Point  ...   POINT Z (154117.981 35706.001 0.000)
            3      Point  ...   POINT Z (152838.030 38773.046 0.000)
            4      Point  ...   POINT Z (155501.881 37036.022 0.000)
            [5 rows x 50 columns]
        """

        path_to_dir = cd(self.DATA_DIR, self.WM_DIRNAME)
        dgn_filename = "CNM_ADMIN.NetworkWaymarks.dgn"
        path_to_pkl = cd(path_to_dir, dgn_filename.replace(".dgn", ".pkl"))

        if os.path.isfile(path_to_pkl) and not update:
            waymarks_shp = load_data(path_to_pkl)

        else:
            if verbose:
                print(f"Reading the shapefile of {self.WM_DIRNAME.lower()}", end=" ... ")

            try:
                waymarks_shp = read_dgn_shapefile(path_to_dir, dgn_filename)

                if verbose:
                    print("Done.")

                save_data(waymarks_shp, path_to_pkl, verbose=verbose)

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")
                waymarks_shp = None

        return waymarks_shp

    def import_waymarks_shp(self, update=False, confirmation_required=True, verbose=True, **kwargs):
        """
        Import shapefile data (converted from the DGN data) of waymarks into the project database.

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

            >>> from src.preprocessor import CNM
            >>> cnm = CNM()
            >>> cnm.import_waymarks_shp(if_exists='replace')
            To import shapefile of waymarks into the table "CNM"."Waymarks"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/cnm_waymarks_shp_tbl.*
            :name: cnm_waymarks_shp_tbl
            :align: center
            :width: 100%

            Snapshot of the "CNM"."Waymarks" table.
        """

        dat_name = f"shapefile of {self.WM_DIRNAME.lower()}"
        tbl_name = f'"{self.SCHEMA_NAME}"."{self.WM_TABLE_NAME}"'

        if confirmed(f"To import {dat_name} into the table {tbl_name}?\n", confirmation_required):
            waymarks_shp = self.read_waymarks_shp(update=update, verbose=verbose)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing the data", end=" ... ")

            try:
                shp_dat = waymarks_shp['Point']

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)

                    if shp_dat is not None:
                        shp_dat['geometry'] = shp_dat['geometry'].map(lambda x: x.wkt)

                self.db_instance.import_data(
                    data=shp_dat, schema_name=self.SCHEMA_NAME, table_name=self.WM_TABLE_NAME,
                    method=self.db_instance.psql_insert_copy, confirmation_required=False,
                    **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")

    def map_view(self, item_name='Waymarks', layer_name='Point', desc_col_name='ASSETID',
                 sample=True, marker_colour='purple', update=False, verbose=True):
        """
        Make a map view of a given item.

        :param item_name: Name of the item; defaults to ``'Waymarks'``.
        :type item_name: str
        :param layer_name: Name of the layer; defaults to ``'Point'``.
        :type layer_name: str
        :param desc_col_name: Name of the column that describes markers; defaults to ``'ASSETID'``.
        :type desc_col_name: str
        :param sample: Whether to draw a sample or a specific sample size; defaults to ``True``.
        :type sample: bool | int
        :param marker_colour: Colour of markers; defaults to ``'purple'``.
        :type marker_colour: str
        :param update: Whether to reprocess the original data files; defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int

        **Examples**::

            >>> from src.preprocessor import CNM
            >>> cnm = CNM()
            >>> cnm.map_view(desc_col_name='ASSETID', sample=100)

        .. raw:: html

            <iframe src="../_static/view_waymarks.html" marginwidth="0"
                marginheight="0" scrolling="no"
                style="width:100%; height:600px; border:0; overflow:hidden;">
            </iframe>

        .. only:: html

            .. figure:: ../_images/blank.png
                :name: blank_view_waymarks
                :align: center
                :width: 0%

                Examples of waymarks.

        .. only:: latex

            .. figure:: ../_images/cnm_waymarks_view_demo.*
                :name: cnm_waymarks_view_demo
                :align: center
                :width: 100%

                Examples of waymarks.
        """

        dgn_shp_map_view(
            self, item_name=item_name, layer_name=layer_name, desc_col_name=desc_col_name,
            sample=sample, marker_colour=marker_colour, update=update, verbose=verbose)

    def load_waymarks_shp(self, elr=None, column_names=None, fmt_nr_mileage=False, **kwargs):
        """
        Load the points layer of the waymarks shapefile (for a given ELR or ELRs)
        from the project database.

        :param elr: Engineer's Line Reference(s); defaults to ``None``.
        :type elr: str | list | tuple | None
        :param column_names: Names of (a subset of) columns to be queried; defaults to ``None``.
        :type column_names: str | list | None
        :param fmt_nr_mileage: Whether to add formatted Network Rail mileage data;
            defaults to ``False``.
        :type fmt_nr_mileage: bool
        :param kwargs: [Optional] parameters for the method
            `pyhelpers.dbms.PostgreSQL.read_sql_query`_.
        :return: Data of the waymarks shapefile.
        :rtype: pandas.DataFrame

        .. _`pyhelpers.dbms.PostgreSQL.read_sql_query`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.read_sql_query.html

        **Examples**::

            >>> from src.preprocessor import CNM
            >>> cnm = CNM()
            >>> waymarks_points = cnm.load_waymarks_shp()
            >>> waymarks_points.head()
                  Entity  ...                                           geometry
            0      Point  ...  POINT Z (492436.76070000045 168198.35700000077 0)
            1      Point  ...   POINT Z (491806.33029999956 168202.1041000001 0)
            2      Point  ...  POINT Z (491737.74010000005 167755.89090000093 0)
            3      Point  ...  POINT Z (491787.06960000005 167357.06870000064 0)
            4      Point  ...   POINT Z (491888.97809999995 166960.9595999997 0)
            [5 rows x 50 columns]
            >>> waymarks_points = cnm.load_waymarks_shp(elr='ECM8')
            >>> waymarks_points.head()
                Entity  ...                                         geometry
            0    Point  ...  POINT Z (327289.9845000003 674251.0335000008 0)
            1    Point  ...  POINT Z (327696.2715999996 674194.5047999993 0)
            2    Point  ...  POINT Z (328097.8353000004 674240.6223000009 0)
            3    Point  ...  POINT Z (328494.3197999997 674324.2128999997 0)
            4    Point  ...  POINT Z (328893.0425000004 674278.1779999994 0)
            [5 rows x 51 columns]
            >>> waymarks_points = cnm.load_waymarks_shp(['ECM7', 'ECM8'], column_names='essential')
            >>> waymarks_points.head()
                  ELR  WAYMARK_VALUE                                         geometry
            0    ECM7         0.0000  POINT Z (424615.7734000003 563820.6427999996 0)
            1    ECM7         0.0440  POINT Z (425015.6885000002 563885.4956999999 0)
            2    ECM7         0.0880  POINT Z (425266.2346999999 564189.9217000008 0)
            3    ECM7         0.1320  POINT Z (425568.8497000001 564439.0111999996 0)
            4    ECM7         1.0000  POINT Z (425931.0060999999 564623.0390000008 0)
            [5 rows x 3 columns]
        """

        essential_columns = ['ELR', 'WAYMARK_VALUE', 'geometry']
        if column_names is not None:
            if column_names == 'essential':
                column_names_ = essential_columns.copy()
            else:
                assert isinstance(column_names, (list, tuple))
                assert all(col in essential_columns for col in column_names)
                column_names_ = copy.copy(column_names)
        else:
            column_names_ = column_names

        column_names_ = validate_column_names(column_names_)
        sql_query = f'SELECT {column_names_} FROM "{self.SCHEMA_NAME}"."{self.WM_TABLE_NAME}"'

        sql_query = add_sql_query_elr_condition(sql_query, elr=elr)

        if column_names_ == '*' or 'ASSETID' in column_names_:
            sort_by_cols = ['ASSETID']
        else:
            sort_by_cols = essential_columns[:2]

        if self.db_instance is None:
            self.db_instance = TrackFixityDB()

        waymarks_shp = self.db_instance.read_sql_query(sql_query=sql_query, **kwargs).sort_values(
            sort_by_cols, ignore_index=True)

        waymarks_shp.geometry = waymarks_shp.geometry.map(shapely.wkt.loads)

        if fmt_nr_mileage:
            waymarks_shp['Mileage'] = waymarks_shp.WAYMARK_VALUE.map(mileage_num_to_str)

        return waymarks_shp

    @staticmethod
    def make_pseudo_waymarks(waymarks):
        # noinspection PyShadowingNames
        """
        Make pseudo mileages and geometry objects of waymarks.

        :param waymarks: Data of waymarks.
        :type waymarks: pandas.DataFrame
        :return: Data of waymarks with pseudo mileages.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import CNM
            >>> cnm = CNM()
            >>> waymarks = cnm.load_waymarks_shp(['ECM7', 'ECM8'], column_names='essential')
            >>> waymarks_ = cnm.make_pseudo_waymarks(waymarks)
            >>> waymarks_.head()
                  ELR  ...                                    pseudo_geometry
            0    ECM7  ...  LINESTRING Z (424615.7734000003 563820.6427999...
            1    ECM7  ...  LINESTRING Z (425015.68850000016 563885.495699...
            2    ECM7  ...  LINESTRING Z (425266.2346999999 564189.9217000...
            3    ECM7  ...  LINESTRING Z (425568.84970000014 564439.011199...
            4    ECM7  ...  LINESTRING Z (425931.0060999999 564623.0390000...
            [5 rows x 6 columns]
        """

        if 'WAYMARK_VALUE' in waymarks.columns:
            waymarks_ = waymarks.rename(columns={'WAYMARK_VALUE': 'Mileage'})
        else:
            waymarks_ = waymarks.copy()

        waymarks_ = waymarks_.join(waymarks_['Mileage'].shift(-1), rsuffix='_end')
        end_val = pd.Series(
            waymarks_['Mileage'].iloc[-1],
            index=waymarks_.index[waymarks_['Mileage_end'].isnull()])
        waymarks_['Mileage_end'] = waymarks_['Mileage_end'].fillna(end_val)

        waymarks_ = waymarks_.join(waymarks_['geometry'].shift(-1), rsuffix='_end')
        end_val = pd.Series(
            waymarks_['geometry'].iloc[-1],
            index=waymarks_.index[waymarks_['geometry_end'].isnull()])
        waymarks_['geometry_end'] = waymarks_['geometry_end'].fillna(end_val)

        waymarks_['pseudo_geometry'] = waymarks_.apply(
            lambda x: LineString([x['geometry'], x['geometry_end']]).coords, axis=1).map(
            LineString)

        rename_columns = {
            'Mileage': 'StartMileage',
            'geometry': 'StartMileageGeom',
            'Mileage_end': 'EndMileage',
            'geometry_end': 'EndMileageGeom'}
        waymarks_.rename(columns=rename_columns, inplace=True)

        return waymarks_
