"""
A class for preprocessing data about structures from
*Civil Asset Register and Reporting System (CARRS)*.
"""

import copy
import glob
import os

import numpy as np
import pandas as pd
import shapely.wkt
from pyhelpers._cache import _print_failure_msg
from pyhelpers.dirs import cd, cdd
from pyhelpers.ops import confirmed
from pyhelpers.store import load_data, save_data
from pyhelpers.text import find_similar_str, get_acronym

from src.utils.dgn import dgn2shp_batch, dgn_shp_map_view, get_dgn_shp_prj, read_dgn_shapefile
from src.utils.general import TrackFixityDB, add_sql_query_elr_condition


class CARRS:
    """
    *Civil Asset Register and Reporting System*.

    This class handles preprocessing of summary and DGN data for various types of structures,
    including overline *bridges*, *underline bridges*, *retaining walls* and *tunnels*.

    .. note::

        This class focuses specifically on handling summary and DGN data related to the
        aforementioned structure types.
    """

    #: Data name.
    NAME: str = 'Civil Asset Register and Reporting System'
    #: Acronym of the data name.
    ACRONYM: str = get_acronym(NAME, only_capitals=True)

    #: Pathname of a local directory where the CARRS data is stored.
    DATA_DIR: str = os.path.relpath(cdd(ACRONYM))
    #: Directory name of projection/boundary shapefiles.
    PROJ_DIRNAME: str = "Projection"
    #: Filename of the projection file.
    PROJ_FILENAME: str = "Order_69433_Polygon"
    #: Directory name of *structures* data.
    STRUCT_DIRNAME: str = "Structures"
    #: Directory name of *overline bridges* data.
    OL_BDG_DIRNAME: str = "Overline bridges"
    #: Directory name of *underline bridges* data.
    UL_BDG_DIRNAME: str = "Underline bridges"
    #: Directory name of *retaining walls* data.
    RW_DIRNAME: str = "Retaining walls"
    #: Directory name of *tunnels* data.
    TUNL_DIRNAME: str = "Tunnels"

    #: Name of the schema for storing the CARRS data.
    SCHEMA_NAME: str = copy.copy(ACRONYM)
    #: Name of the table for storing the *structures* data.
    STRUCT_TABLE_NAME: str = STRUCT_DIRNAME.replace(' ', '_')
    #: Name of the table for storing the *overline bridges* data.
    OL_BDG_TABLE_NAME: str = OL_BDG_DIRNAME.replace(' ', '_')
    #: Name of the table for storing the *underline bridges* data.
    UL_BDG_TABLE_NAME: str = UL_BDG_DIRNAME.replace(' ', '_')
    #: Name of the table for storing the *retaining walls* data.
    RW_TABLE_NAME: str = RW_DIRNAME.replace(' ', '_')
    #: Name of the table for storing the *tunnels* data.
    TUNL_TABLE_NAME: str = TUNL_DIRNAME.replace(' ', '_')

    def __init__(self, db_instance=None, elr=None):
        """
        :param db_instance: PostgreSQL database instance; defaults to ``None``.
        :type db_instance: TrackFixityDB | None
        :param elr: Engineer's Line References; defaults to ``None``.
        :type elr: str | list | tuple | None

        :ivar str elr: Engineer's Line References.
        :ivar dict struct_dtypes: Data types of the *structures* data.
        :ivar list ol_bdg_dgn_pathnames: Pathname(s) of DGN data of *overline bridges*.
        :ivar list ul_bdg_dgn_pathnames: Pathname(s) of DGN data of *underline bridges*.
        :ivar list rw_dgn_pathnames: Pathname(s) of DGN data of *retaining walls*.
        :ivar list tunl_dgn_pathnames: Pathname(s) of DGN data of *tunnels*.

        :ivar pyhelpers.dbms.PostgreSQL db_instance: PostgreSQL database instance
            used for data operations.

        **Examples**::

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()
            >>> carrs.NAME
            'Civil Asset Register and Reporting System'

        .. figure:: ../_images/carrs_schema.*
            :name: carrs_schema
            :align: center
            :width: 100%

            Snapshot of the *CARRS* schema.
        """

        super().__init__()

        self.elr = elr

        self.struct_filename = 'Structures'
        self.struct_dtypes = {
            'ELR': str,
            'Start': str,
            'End': str,
            'Asset type': str,
            'Railway id': str,
            'Description': str,
            'Operational status': str,
            'Owner': str,
            'Options': str,
        }

        self.ol_bdg_dgn_pathnames = glob.glob(cd(self.DATA_DIR, self.OL_BDG_DIRNAME, "*.dgn"))
        self.ul_bdg_dgn_pathnames = glob.glob(cd(self.DATA_DIR, self.UL_BDG_DIRNAME, "*.dgn"))
        self.rw_dgn_pathnames = glob.glob(cd(self.DATA_DIR, self.RW_DIRNAME, "*.dgn"))
        self.tunl_dgn_pathnames = glob.glob(cd(self.DATA_DIR, self.TUNL_DIRNAME, "*.dgn"))

        self.db_instance = db_instance

    def read_structures(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read data of structures from a local directory.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of structures from *CARRS*.
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()
            >>> # structures_data = carrs.read_structures(update=True, verbose=True)
            >>> # Parsing data of structures ... Done.
            >>> # Updating "Structures.pkl" at "data\\CARRS\\Structures" ... Done.
            >>> data = carrs.read_structures()
            >>> data
                  ELR    Start End  ... Operational status                     Owner Options
            0    ECM8  10.0004      ...        Functionary             Outside Party
            1    ECM8  10.0266      ...        Functionary  Network Rail (CE-Struct)
            2    ECM8  10.0308      ...        Functionary  Network Rail (CE-Struct)
            3    ECM8  10.0324      ...        Functionary  Network Rail (CE-Struct)
            4    ECM8  10.0330      ...        Functionary  Network Rail (CE-Struct)
            ..    ...      ...  ..  ...                ...                       ...     ...
            195  ECM8  53.0880      ...        Functionary  Network Rail (CE-Struct)
            196  ECM8  53.1320      ...        Functionary  Network Rail (CE-Struct)
            197  ECM8  53.1673      ...        Functionary            NR Maintenance
            198  ECM8  54.0264      ...        Functionary  Network Rail (CE-Struct)
            199  ECM8  54.0286      ...        Functionary  Network Rail (CE-Struct)
            [200 rows x 9 columns]
        """

        path_to_pickle = cd(self.DATA_DIR, self.STRUCT_DIRNAME, self.struct_filename + ".pkl")

        if os.path.isfile(path_to_pickle) and not update:
            data = load_data(path_to_pickle)

        else:
            if verbose:
                print("Parsing data of {}".format(self.STRUCT_DIRNAME.lower()), end=" ... ")

            path_to_file = path_to_pickle.replace(".pkl", ".xlsx")
            try:
                data = pd.read_excel(path_to_file, dtype=self.struct_dtypes).replace({np.nan: ''})

                data.replace({np.nan: ''}, inplace=True)

                if verbose:
                    print("Done.")

                save_data(data, path_to_pickle, verbose=verbose)

            except Exception as e:
                _print_failure_msg(e)

                data = None

        return data

    def import_structures(self, update=False, confirmation_required=True, verbose=True, **kwargs):
        """
        Import data of structures into the project database.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param confirmation_required: Whether confirmation is required to proceed;
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

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()
            >>> carrs.import_structures(if_exists='replace')
            To import data of structures into the table "CARRS"."Structures"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/carrs_structures_tbl.*
            :name: carrs_structures_tbl
            :align: center
            :width: 100%

            Snapshot of the "CARRS"."Structures" table.
        """

        dat_name = f"data of {self.STRUCT_DIRNAME.lower()}"
        tbl_name = f"\"{self.SCHEMA_NAME}\".\"{self.STRUCT_TABLE_NAME}\""

        if confirmed(f"To import {dat_name} into the table {tbl_name}?\n", confirmation_required):
            data = self.read_structures(update=update, verbose=verbose)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing the data", end=" ... ")

            try:
                self.db_instance.import_data(
                    data=data, schema_name=self.SCHEMA_NAME, table_name=self.STRUCT_TABLE_NAME,
                    method=self.db_instance.psql_insert_copy, confirmation_required=False,
                    **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")

    def read_prj_metadata(self, update=False, parser='osr', as_dict=True, verbose=True):
        # noinspection PyShadowingNames
        """
        Read metadata of projection for the DGN files/shapefiles.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param parser: Name of package used for reading the PRJ file;
            options include ``{'osr', 'pycrs'}``; defaults to ``'osr'``.
        :type parser: str
        :param as_dict: Whether to return the data as a dictionary; defaults to ``True``.
        :type as_dict: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``True``.
        :type verbose: bool | int
        :return: Metadata of the projection for the DGN files/shapefiles.
        :rtype: dict | None

        **Examples**::

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()
            >>> dgn_shp_prj = carrs.read_prj_metadata()
            >>> type(dgn_shp_prj)
            dict
            >>> list(dgn_shp_prj.keys())
            ['PROJCS', 'Shapefile']
            >>> type(dgn_shp_prj['PROJCS'])
            dict
            >>> list(dgn_shp_prj['PROJCS'].keys())
            ['proj', 'lat_0', 'lon_0', 'k', 'x_0', 'y_0', 'ellps', 'units']
            >>> dgn_shp_prj['Shapefile']
              ORDER_ID  ...                                           geometry
            0    69433  ...  POLYGON ((307732.043 1047233.175, 702872.281 4...
            [1 rows x 4 columns]
        """

        path_to_file = cd(self.DATA_DIR, self.PROJ_DIRNAME, self.PROJ_FILENAME)

        dgn_shp_prj = get_dgn_shp_prj(
            path_to_file=path_to_file, update=update, projcs_parser=parser, as_dict=as_dict,
            verbose=verbose)

        return dgn_shp_prj

    def dgn2shp(self, dat_name=None, confirmation_required=True, verbose=True, **kwargs):
        """
        Convert DGN files to shapefiles.

        :param dat_name: Name of the data; defaults to ``None``.
        :type dat_name: str | None
        :param confirmation_required: Whether a confirmation is required to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the
            :func:`~src.utils.dgn.dgn2shp` function.

        **Examples**::

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()
            >>> carrs.dgn2shp()
            To convert all available .dgn files of "CARRS" to shapefiles?
             [No]|Yes: yes
            Converting "... ... _Overline_Bridge.dgn" at "data\\CARRS\\Overline bridges" ... Done.
            Converting "... ... _Underline_Bridge.dgn" at "data\\CARRS\\Underline bridges" ... Done.
            Converting "... ... _Retaining_Wall.dgn" at "data\\CARRS\\Retaining walls" ... Done.
            Converting "... ... _TunnelPortal.dgn" at "data\\CARRS\\Tunnels" ... Done.
        """

        folder_names = (self.OL_BDG_DIRNAME, self.UL_BDG_DIRNAME, self.RW_DIRNAME, self.TUNL_DIRNAME)

        if dat_name in folder_names:
            dgn2shp_batch(
                dat_name=dat_name, file_paths=glob.glob(cd(self.DATA_DIR, dat_name, "*.dgn")),
                confirmation_required=confirmation_required, verbose=verbose,
                **kwargs)

        elif dat_name is None:
            cfm_msg = f"To convert all available .dgn files of \"{self.ACRONYM}\" to shapefiles?\n"
            if confirmed(cfm_msg, confirmation_required=confirmation_required):
                paths_to_dgn_files = glob.glob(cd(self.DATA_DIR, "*", "*.dgn"))

                dgn2shp_batch(
                    self.ACRONYM, paths_to_dgn_files, confirmation_required=False, verbose=verbose,
                    **kwargs)

    def _read_dgn_shp(self, structure_name, dgn_filename, update=False, verbose=False):
        """
        Read the shapefile (converted from the DGN data) of an asset from a local directory.

        :param structure_name: Name of the asset.
        :type structure_name: str
        :param dgn_filename: Filename of the DGN data of the asset.
        :type dgn_filename: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of the DGN-converted shapefile of the asset.
        :rtype: dict | None

        .. seealso::

            - Examples of the following methods:
              - :meth:`~src.preprocessor.CARRS.read_overline_bridges_shp`
              - :meth:`~src.preprocessor.CARRS.read_underline_bridges_shp`
              - :meth:`~src.preprocessor.CARRS.read_retaining_walls_shp`
              - :meth:`~src.preprocessor.CARRS.read_tunnels_shp`
        """

        data_names = [self.__getattribute__(x) for x in dir(self) if x.endswith('_DIRNAME')]
        structure_name_ = find_similar_str(structure_name, data_names)

        path_to_dir = cd(self.DATA_DIR, structure_name_)
        path_to_pickle = cd(path_to_dir, dgn_filename.replace(".dgn", ".pkl"))

        if os.path.isfile(path_to_pickle) and not update:
            asset_dgn_shp = load_data(path_to_pickle)

        else:
            if verbose:
                print(f"Reading the shapefile of {structure_name_.lower()}", end=" ... ")

            try:
                asset_dgn_shp = read_dgn_shapefile(path_to_dir, dgn_filename)

                if verbose:
                    print("Done.")

                save_data(asset_dgn_shp, path_to_file=path_to_pickle, verbose=verbose)

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")

                asset_dgn_shp = None

        return asset_dgn_shp

    def _import_dgn_shp(self, structure_name, update=False, confirmation_required=True,
                        verbose=True, **kwargs):
        """
        Import shapefile data (converted from the DGN data) of an asset into the project database.

        :param structure_name: Name of the asset.
        :type structure_name: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param confirmation_required: Whether confirmation is required to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the
            `pyhelpers.dbms.PostgreSQL.import_data`_ method.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        .. seealso::

            - Examples of the following methods:
              - :meth:`~src.preprocessor.carrs.CARRS.import_overline_bridges_shp`
              - :meth:`~src.preprocessor.carrs.CARRS.import_underline_bridges_shp`
              - :meth:`~src.preprocessor.carrs.CARRS.import_retaining_walls_shp`
              - :meth:`~src.preprocessor.carrs.CARRS.import_tunnels_shp`
        """

        data_names = [self.__getattribute__(x) for x in dir(self) if x.endswith('_DIRNAME')]
        structure_name_ = find_similar_str(structure_name, data_names).lower()
        dat_name = f"the DGN shapefile of {structure_name_}"

        table_names = [self.__getattribute__(x) for x in dir(self) if x.endswith('_TABLE_NAME')]
        table_name = find_similar_str(structure_name, table_names)
        tbl_name = f"\"{self.SCHEMA_NAME}\".\"{table_name}\""

        if confirmed(f"To import {dat_name} into {tbl_name}?\n", confirmation_required):

            func_name = 'read_{}_shp'.format(structure_name_.replace(' ', '_'))
            asset_dgn_shp = self.__getattribute__(func_name)(update=update, verbose=verbose)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                if not confirmation_required:
                    print(f"Importing the {dat_name}", end=" ... ")
                else:
                    print("Importing the data", end=" ... ")

            try:
                shp_dat = asset_dgn_shp['Point']

                if shp_dat is not None:
                    shp_dat['geometry'] = shp_dat['geometry'].map(lambda x: x.wkt)

                self.db_instance.import_data(
                    data=shp_dat, schema_name=self.SCHEMA_NAME, table_name=table_name,
                    method=self.db_instance.psql_insert_copy, confirmation_required=False,
                    **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")

    def _load_dgn_shp(self, structure_name, sql_query=None, elr=None, **kwargs):
        """
        Load the shapefile data (converted from the DGN data) of an asset from the project database.

        :param structure_name: Name of the asset.
        :type structure_name: str
        :param sql_query: PostgreSQL query statement; defaults to ``None``.
        :type sql_query: str | None
        :param elr: Engineer's Line Reference(s); defaults to ``None``.
        :type elr: str | list | tuple | None
        :param kwargs: [Optional] additional parameters for the
            method `pyhelpers.dbms.PostgreSQL.read_sql_query`_.
        :return: Data of the shapefile (converted from the DGN data) of the asset.
        :rtype: pandas.DataFrame

        .. _`pyhelpers.dbms.PostgreSQL.read_sql_query`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.read_sql_query.html

        .. seealso::

            - Examples of the following methods:
              - :meth:`~src.preprocessor.CARRS.load_overline_bridges_shp`
              - :meth:`~src.preprocessor.CARRS.load_underline_bridges_shp`
              - :meth:`~src.preprocessor.CARRS.load_retaining_walls_shp`
              - :meth:`~src.preprocessor.CARRS.load_tunnels_shp`
        """

        if self.db_instance is None:
            self.db_instance = TrackFixityDB()

        table_names = [
            self.__getattribute__(x).replace(" ", "_") for x in dir(self) if x.endswith("_DIRNAME")]
        tbl_name = find_similar_str(structure_name, table_names)

        if sql_query is None:
            sql_qry = f'SELECT * FROM "{self.SCHEMA_NAME}"."{tbl_name}"'
            sql_qry = add_sql_query_elr_condition(sql_qry, elr=elr)
            sql_qry += ' ORDER BY "ELR" ASC, "RAILWAY_ID" ASC'
        else:
            sql_qry = sql_query

        dgn_shp_dat = self.db_instance.read_sql_query(sql_query=sql_qry, **kwargs)

        if 'geometry' in dgn_shp_dat.columns:
            dgn_shp_dat['geometry'] = dgn_shp_dat['geometry'].map(shapely.wkt.loads)

        return dgn_shp_dat

    def read_overline_bridges_shp(self, **kwargs):
        # noinspection PyShadowingNames
        """
        Read the shapefile (converted from the DGN data) of *overline bridges*
        from a local directory.

        :param kwargs: Parameters of the method :meth:`src.preprocessor.carrs.CARRS._read_dgn_shp`.
        :return: Data of the shapefile of *overline bridges*.
        :rtype: dict | None

        **Examples**::

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()
            >>> ol_bdg_shp = carrs.read_overline_bridges_shp()
            >>> type(ol_bdg_shp)
            dict
            >>> list(ol_bdg_shp.keys())
            ['Annotation', 'MultiPatch', 'Point', 'Polygon', 'Polyline']
            >>> ol_bdg_shp['MultiPatch'].empty
            True
            >>> ol_bdg_shp['Annotation'].empty
            True
            >>> ol_bdg_shp['Point'].empty
            False
            >>> ol_bdg_shp['Polygon'].empty
            True
            >>> ol_bdg_shp['Polyline'].empty
            True
            >>> ol_bdg_shp['Point'].head()
                  Entity  ...                               geometry
            0      Point  ...   POINT Z (150671.813 31303.480 0.000)
            1      Point  ...   POINT Z (151667.807 31945.272 0.000)
            2      Point  ...   POINT Z (152952.274 33955.715 0.000)
            3      Point  ...   POINT Z (156042.572 37403.277 0.000)
            4      Point  ...   POINT Z (158027.581 38004.746 0.000)
            [5 rows x 78 columns]
        """

        ol_bdg_shp = self._read_dgn_shp(
            structure_name=self.OL_BDG_DIRNAME, dgn_filename="ENG_ADMIN.CARRS_Overline_Bridge.dgn",
            **kwargs)

        return ol_bdg_shp

    def import_overline_bridges_shp(self, **kwargs):
        """
        Import shapefile data (converted from the DGN data) of *overline bridges*
        into the project database.

        :param kwargs: Parameters of the method :meth:`src.preprocessor.CARRS._import_dgn_shp`.

        **Examples**::

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()
            >>> carrs.import_overline_bridges_shp(if_exists='replace')
            To import the DGN shapefile of overline bridges into "CARRS"."Overline_bridges"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/carrs_overline_bridges_shp_tbl.*
            :name: carrs_overline_bridges_shp_tbl
            :align: center
            :width: 100%

            Snapshot of the "CARRS"."Overline_bridges" table.
        """

        self._import_dgn_shp(structure_name=self.OL_BDG_TABLE_NAME, **kwargs)

    def load_overline_bridges_shp(self, **kwargs):
        # noinspection PyShadowingNames
        """
        Load the shapefile data (converted from the DGN data) of *overline bridges*
        from the project database.

        :param kwargs: Parameters of the method :meth:`src.preprocessor.carrs.CARRS._load_dgn_shp`.
        :return: Data of the shapefile (converted from the DGN data) of *overline bridges*.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()
            >>> ol_bdg_shp = carrs.load_overline_bridges_shp(elr=['ECM7', 'ECM8'])
            >>> ol_bdg_shp.head()
                Entity  ...                                         geometry
            0    Point  ...  POINT Z (423181.7536000004 610967.6328999996 0)
            1    Point  ...  POINT Z (423785.0826000003 613768.3622999992 0)
            2    Point  ...        POINT Z (423624.1886 616374.8772999998 0)
            3    Point  ...        POINT Z (423184.8782000002 616951.3377 0)
            4    Point  ...        POINT Z (421395.7363 620237.2636999991 0)
            [5 rows x 78 columns]
        """

        ol_bdg_shp = self._load_dgn_shp(structure_name=self.OL_BDG_TABLE_NAME, **kwargs)

        return ol_bdg_shp

    def read_underline_bridges_shp(self, **kwargs):
        # noinspection PyShadowingNames
        """
        Read the shapefile (converted from the DGN data) of *underline bridges*
        from a local directory.

        :param kwargs: Parameters of the method :meth:`src.preprocessor.carrs.CARRS._read_dgn_shp`.
        :return: Data of the shapefile of *underline bridges*.
        :rtype: dict | None

        **Examples**::

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()
            >>> ul_bdg_shp = carrs.read_underline_bridges_shp()
            >>> type(ul_bdg_shp)
            dict
            >>> list(ul_bdg_shp.keys())
            ['Annotation', 'MultiPatch', 'Point', 'Polygon', 'Polyline']
            >>> ul_bdg_shp['Annotation'].empty
            True
            >>> ul_bdg_shp['MultiPatch'].empty
            True
            >>> ul_bdg_shp['Point'].empty
            False
            >>> ul_bdg_shp['Polygon'].empty
            True
            >>> ul_bdg_shp['Polyline'].empty
            True
            >>> ul_bdg_shp['Point'].head()
                  Entity  ...                               geometry
            0      Point  ...   POINT Z (153777.071 35335.486 0.000)
            1      Point  ...   POINT Z (154081.346 35669.935 0.000)
            2      Point  ...   POINT Z (162641.973 39207.381 0.000)
            3      Point  ...   POINT Z (166286.374 40370.296 0.000)
            4      Point  ...   POINT Z (166318.680 40394.695 0.000)
            [5 rows x 78 columns]
        """

        ul_bdg_shp = self._read_dgn_shp(
            structure_name=self.UL_BDG_DIRNAME, dgn_filename="ENG_ADMIN.CARRS_Underline_Bridge.dgn",
            **kwargs)

        return ul_bdg_shp

    def import_underline_bridges_shp(self, **kwargs):
        """
        Import shapefile data (converted from the DGN data) of *underline bridges*
        into the project database.

        :param kwargs: Parameters of the method :meth:`src.preprocessor.CARRS._import_dgn_shp`.

        **Examples**::

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()
            >>> carrs.import_underline_bridges_shp(if_exists='replace')
            To import the DGN shapefile of underline bridges into "CARRS"."Underline_bridges"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/carrs_underline_bridges_shp_tbl.*
            :name: carrs_underline_bridges_shp_tbl
            :align: center
            :width: 100%

            Snapshot of the "CARRS"."Underline_bridges" table.
        """

        self._import_dgn_shp(structure_name=self.UL_BDG_TABLE_NAME, **kwargs)

    def load_underline_bridges_shp(self, **kwargs):
        # noinspection PyShadowingNames
        """
        Load the shapefile data (converted from the DGN data) of *underline bridges*
        from the project database.

        :param kwargs: Parameters of the method :meth:`src.preprocessor.carrs.CARRS._load_dgn_shp`.
        :return: Data of the shapefile (converted from the DGN data) of underline bridges.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()
            >>> ul_bdg_shp = carrs.load_underline_bridges_shp(elr=['ECM7', 'ECM8'])
            >>> ul_bdg_shp.head()
                Entity  ...                                         geometry
            0    Point  ...  POINT Z (424644.8838999998 563787.7875999995 0)
            1    Point  ...  POINT Z (424644.8838999998 563787.7875999995 0)
            2    Point  ...  POINT Z (423963.1664000005 609154.5935999993 0)
            3    Point  ...  POINT Z (423856.8586999997 609453.7774999999 0)
            4    Point  ...  POINT Z (423721.8256000001 609828.4415000007 0)
            [5 rows x 78 columns]
        """

        ul_bdg_shp = self._load_dgn_shp(structure_name=self.UL_BDG_TABLE_NAME, **kwargs)

        return ul_bdg_shp

    def read_retaining_walls_shp(self, **kwargs):
        # noinspection PyShadowingNames
        """
        Read the shapefile (converted from the DGN data) of *retaining walls*
        from a local directory.

        :param kwargs: Parameters of the method :meth:`src.preprocessor.carrs.CARRS._read_dgn_shp`.
        :return: Data of the shapefile of *retaining walls*.
        :rtype: dict | None

        **Examples**::

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()
            >>> retg_walls_shp = carrs.read_retaining_walls_shp()
            >>> type(retg_walls_shp)
            dict
            >>> list(retg_walls_shp.keys())
            ['Annotation', 'MultiPatch', 'Point', 'Polygon', 'Polyline']
            >>> retg_walls_shp['Annotation'].empty
            True
            >>> retg_walls_shp['MultiPatch'].empty
            True
            >>> retg_walls_shp['Point'].empty
            False
            >>> retg_walls_shp['Polygon'].empty
            True
            >>> retg_walls_shp['Polyline'].empty
            True
            >>> retg_walls_shp['Point'].head()
                  Entity  ...                               geometry
            0      Point  ...   POINT Z (152580.106 33145.415 0.000)
            1      Point  ...   POINT Z (153723.038 35252.128 0.000)
            2      Point  ...   POINT Z (152157.389 39942.775 0.000)
            3      Point  ...   POINT Z (152344.545 39897.870 0.000)
            4      Point  ...   POINT Z (152531.952 39017.593 0.000)
            [5 rows x 60 columns]
        """

        retg_walls_shp = self._read_dgn_shp(
            structure_name=self.RW_DIRNAME, dgn_filename="ENG_ADMIN.CARRS_Retaining_Wall.dgn",
            **kwargs)

        return retg_walls_shp

    def import_retaining_walls_shp(self, **kwargs):
        """
        Import shapefile data (converted from the DGN data) of *retaining walls*
        into the project database.

        :param kwargs: Parameters of the method :meth:`src.preprocessor.CARRS._import_dgn_shp`.

        **Examples**::

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()
            >>> carrs.import_retaining_walls_shp(if_exists='replace')
            To import shapefile of retaining walls into the table "CARRS"."Retaining_walls"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/carrs_retaining_walls_shp_tbl.*
            :name: carrs_retaining_walls_shp_tbl
            :align: center
            :width: 100%

            Snapshot of the "CARRS"."Retaining_walls" table.
        """

        self._import_dgn_shp(structure_name=self.RW_TABLE_NAME, **kwargs)

    def load_retaining_walls_shp(self, **kwargs):
        # noinspection PyShadowingNames
        """
        Load the shapefile data (converted from the DGN data) of *retaining walls*
        from the project database.

        :param kwargs: Parameters of the method :meth:`src.preprocessor.carrs.CARRS._load_dgn_shp`.
        :return: Data of the shapefile (converted from the DGN data) of *retaining walls*.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()
            >>> retg_walls_shp = carrs.load_retaining_walls_shp(elr=['ECM7', 'ECM8'])
            >>> retg_walls_shp.head()
                Entity  ...                                         geometry
            0    Point  ...  POINT Z (425355.5612000003 564306.7263999991 0)
            1    Point  ...  POINT Z (425448.4572000001 564368.4912999999 0)
            2    Point  ...        POINT Z (425454.0697999997 564371.4035 0)
            3    Point  ...  POINT Z (425467.7002999997 564378.4759999998 0)
            4    Point  ...  POINT Z (425483.8436000003 564386.5826999992 0)
            [5 rows x 60 columns]
        """

        retg_walls_shp = self._load_dgn_shp(structure_name=self.RW_TABLE_NAME, **kwargs)

        return retg_walls_shp

    def read_tunnels_shp(self, **kwargs):
        # noinspection PyShadowingNames
        """
        Read the shapefile (converted from the DGN data) of *tunnels* from a local directory.

        :param kwargs: Parameters of the method :meth:`src.preprocessor.carrs.CARRS._read_dgn_shp`.
        :return: Data of the shapefile (converted from the DGN data) of *tunnels*.
        :rtype: dict | None

        **Examples**::

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()
            >>> tunl_shp = carrs.read_tunnels_shp()
            >>> type(tunl_shp)
            dict
            >>> list(tunl_shp.keys())
            ['Annotation', 'MultiPatch', 'Point', 'Polygon', 'Polyline']
            >>> tunl_shp['Annotation'].empty
            True
            >>> tunl_shp['MultiPatch'].empty
            True
            >>> tunl_shp['Point'].empty
            False
            >>> tunl_shp['Polygon'].empty
            True
            >>> tunl_shp['Polyline'].empty
            True
            >>> tunl_shp['Point'].head()
                 Entity  ...                               geometry
            0     Point  ...  POINT Z (175570.798 782711.751 0.000)
            1     Point  ...  POINT Z (185539.688 781374.833 0.000)
            2     Point  ...  POINT Z (191942.787 780080.803 0.000)
            3     Point  ...  POINT Z (221042.577 654746.431 0.000)
            4     Point  ...  POINT Z (227213.665 676197.641 0.000)
            [5 rows x 69 columns]
        """

        tunl_shp = self._read_dgn_shp(
            structure_name=self.TUNL_DIRNAME, dgn_filename="ENG_ADMIN.RINM_CARRS_TunnelPortal.dgn",
            **kwargs)

        return tunl_shp

    def import_tunnels_shp(self, **kwargs):
        """
        Import shapefile data (converted from the DGN data) of *tunnels*
        into the project database.

        :param kwargs: Parameters of the method :meth:`src.preprocessor.CARRS._import_dgn_shp`.

        **Examples**::

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()
            >>> carrs.import_tunnels_shp(if_exists='replace')
            To import shapefile of tunnels into the table "CARRS"."Tunnels"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/carrs_tunnels_shp_tbl.*
            :name: carrs_tunnels_shp_tbl
            :align: center
            :width: 100%

            Snapshot of the "CARRS"."Tunnels" table.
        """

        self._import_dgn_shp(structure_name=self.TUNL_TABLE_NAME, **kwargs)

    def load_tunnels_shp(self, **kwargs):
        # noinspection PyShadowingNames
        """
        Load the shapefile data (converted from the DGN data) of *tunnels*
        from the project database.

        :param kwargs: Parameters of the method :meth:`src.preprocessor.carrs.CARRS._load_dgn_shp`.
        :return: Data of the shapefile (converted from the DGN data) of *tunnels*.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()
            >>> tunl_shp = carrs.load_tunnels_shp(elr=['ECM7', 'ECM8'])
            >>> tunl_shp.head()
                Entity  ...                                         geometry
            0    Point  ...  POINT Z (425355.5612000003 564306.7263999991 0)
            1    Point  ...  POINT Z (425448.4572000001 564368.4912999999 0)
            2    Point  ...        POINT Z (425454.0697999997 564371.4035 0)
            3    Point  ...  POINT Z (425467.7002999997 564378.4759999998 0)
            4    Point  ...  POINT Z (425483.8436000003 564386.5826999992 0)
            [5 rows x 60 columns]
        """

        tunl_shp = self._load_dgn_shp(structure_name=self.TUNL_TABLE_NAME, **kwargs)

        return tunl_shp

    def map_view(self, structure_name, desc_col_name='DESCRIPTION', sample=True,
                 marker_colour='blue', layer_name='Point', update=False, verbose=True):
        """
        Make a map view of a CARRS item.

        :param structure_name: Name of an item;
            options include ``{'overline bridge', 'underline bridge', 'retaining wall', 'tunnel'}``.
        :type structure_name: str
        :param desc_col_name: Name of a column that describes markers;
            defaults to ``'DESCRIPTION'``.
        :type desc_col_name: str
        :param sample: Whether to draw a sample, or a given sample size; defaults to ``True``.
        :type sample: bool | int
        :param marker_colour: Colour of markers; defaults to ``'blue'``.
        :type marker_colour: str
        :param layer_name: Name of a layer; defaults to ``'Point'``.
        :type layer_name: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int

        **Examples**::

            >>> from src.preprocessor import CARRS
            >>> carrs = CARRS()

        **Overline bridges**::

            >>> # # 100 random examples:
            >>> carrs.map_view('Overline bridges', sample=100, marker_colour='blue')

        .. raw:: html

            <iframe src="../_static/view_overline_bridges.html" marginwidth="0"
                marginheight="0" scrolling="no"
                style="width:100%; height:600px; border:0; overflow:hidden;">
            </iframe>

        .. only:: html

            .. figure:: ../_images/blank.png
                :name: blank_view_overline_bridges
                :align: center
                :width: 0%

                Examples of overline bridges.

        .. only:: latex

            .. figure:: ../_images/carrs_overline_bridges_view_demo.*
                :name: carrs_overline_bridges_view_demo
                :align: center
                :width: 100%

                Examples of overline bridges.

        **Underline bridges**::

            >>> # 100 random examples:
            >>> carrs.map_view('Underline bridges', sample=100, marker_colour='orange')

        .. raw:: html

            <iframe src="../_static/view_underline_bridges.html" marginwidth="0"
                marginheight="0" scrolling="no"
                style="width:100%; height:600px; border:0; overflow:hidden;">
            </iframe>

        .. only:: html

            .. figure:: ../_images/blank.png
                :name: blank_view_underline_bridges
                :align: center
                :width: 0%

                Examples of underline bridges.

        .. only:: latex

            .. figure:: ../_images/carrs_underline_bridges_view_demo.*
                :name: carrs_underline_bridges_view_demo
                :align: center
                :width: 100%

                Examples of underline bridges.

        **Retaining walls**::

            >>> # 100 random examples:
            >>> carrs.map_view('Retaining walls', 'ASSET_DESCRIPTION', 100, marker_colour='green')

        .. raw:: html

            <iframe src="../_static/view_retaining_walls.html" marginwidth="0"
                marginheight="0" scrolling="no"
                style="width:100%; height:600px; border:0; overflow:hidden;">
            </iframe>

        .. only:: html

            .. figure:: ../_images/blank.png
                :name: blank_view_retaining_walls
                :align: center
                :width: 0%

                Random examples of retaining walls.

        .. only:: latex

            .. figure:: ../_images/carrs_retaining_walls_view_demo.*
                :name: carrs_retaining_walls_view_demo
                :align: center
                :width: 100%

                Random examples of retaining walls.

        **Tunnels**::

            >>> # 100 random examples:
            >>> carrs.map_view('Tunnels', sample=100, marker_colour='lightred')

        .. raw:: html

            <iframe src="../_static/view_tunnels.html" marginwidth="0"
                marginheight="0" scrolling="no"
                style="width:100%; height:600px; border:0; overflow:hidden;">
            </iframe>

        .. only:: html

            .. figure:: ../_images/blank.png
                :name: blank_view_tunnels
                :align: center
                :width: 0%

                Random examples of tunnels.

        .. only:: latex

            .. figure:: ../_images/carrs_tunnels_view_demo.*
                :name: carrs_tunnels_view_demo
                :align: center
                :width: 100%

                Random examples of tunnels.
        """

        dgn_shp_map_view(
            self, item_name=structure_name, desc_col_name=desc_col_name, sample=sample,
            marker_colour=marker_colour, layer_name=layer_name, update=update, verbose=verbose)


# if __name__ == '__main__':
#     carrs = CARRS()
#
#     structures_data = carrs.read_structures(update=True, verbose=True)
#
#     carrs.import_structures(if_exists='replace', confirmation_required=False, verbose=True)
#
#     dgn_shp_prj = carrs.read_prj_metadata(update=True, verbose=True)
#
#     map_view_args = {
#         'sample': 100,
#         'update': True,
#         'verbose': True,
#     }
#
#     carrs.map_view(structure_name='Overline bridges', marker_colour='blue', **map_view_args)
# 
#     carrs.map_view(structure_name='Underline bridges', marker_colour='orange', **map_view_args)
#
#     carrs.map_view(
#         structure_name='Retaining walls', desc_col_name='ASSET_DESCRIPTION',
#         marker_colour='green', **map_view_args)
#
#     carrs.map_view(structure_name='Tunnels', marker_colour='lightred', **map_view_args)
