"""
A class for preprocessing data from *Operational Property Asset System (OPAS)*.
"""

import copy
import glob
import os
import warnings

import shapely.wkt
from pyhelpers._cache import _print_failure_msg
from pyhelpers.dirs import cd, cdd
from pyhelpers.ops import confirmed
from pyhelpers.store import load_data, save_data
from pyhelpers.text import get_acronym

from src.utils.dgn import dgn2shp_batch, dgn_shp_map_view, get_dgn_shp_prj, read_dgn_shapefile
from src.utils.general import TrackFixityDB


class OPAS:
    """
    *Operational Property Asset System*.

    .. note::

        This class currently handles only the DGN data about stations.
    """

    #: Data name.
    NAME: str = 'Operational Property Asset System'
    #: Acronym for the data name.
    ACRONYM: str = get_acronym(NAME)
    #: Pathname of a local directory where the OPAS data is stored.
    DATA_DIR: str = os.path.relpath(cdd(ACRONYM))
    #: Directory name of projection/boundary shapefiles.
    PROJ_DIRNAME: str = 'Projection'
    #: Name of the projection file.
    PROJ_FILENAME: str = 'Order_69356_Polygon'
    #: Directory name of *stations* data.
    STN_DIRNAME: str = 'Stations'
    #: Name of the schema for storing the OPAS data.
    SCHEMA_NAME: str = copy.copy(ACRONYM)
    #: Name of the table for storing the *stations* data.
    STN_TABLE_NAME: str = copy.copy(STN_DIRNAME)

    def __init__(self, db_instance=None):
        """
        :param db_instance: PostgreSQL database instance; defaults to ``None``.
        :type db_instance: TrackFixityDB | None

        :ivar TrackFixityDB db_instance: PostgreSQL database instance for database operations.

        **Examples**::

            >>> from src.preprocessor import OPAS
            >>> opas = OPAS()
            >>> opas.NAME
            'Operational Property Asset System'

        .. figure:: ../_images/opas_schema.*
            :name: opas_schema
            :align: center
            :width: 100%

            Snapshot of the *OPAS* schema.
        """

        self.stn_dgn_pathnames = glob.glob(cd(self.DATA_DIR, self.STN_DIRNAME, "*.dgn"))

        self.db_instance = db_instance

    def read_prj_metadata(self, update=False, parser='osr', as_dict=True, verbose=False):
        """
        Read the projection metadata from DGN files or shapefiles.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param parser: Package used for reading the PRJ file;
            options are ``'osr'`` or ``'pycrs'``; defaults to ``'osr'``.
        :type parser: str
        :param as_dict: Whether to return the data as a dictionary; defaults to ``True``.
        :type as_dict: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Projection metadata from the DGN files or shapefiles.
        :rtype: dict | None

        **Examples**::

            >>> from src.preprocessor import OPAS
            >>> opas = OPAS()
            >>> dgn_shp_prj = opas.read_prj_metadata()
            >>> type(dgn_shp_prj)
            dict
            >>> list(dgn_shp_prj.keys())
            ['PROJCS', 'Shapefile']
            >>> list(dgn_shp_prj['PROJCS'].keys())
            ['proj', 'lat_0', 'lon_0', 'k', 'x_0', 'y_0', 'ellps', 'units']
            >>> dgn_shp_prj['Shapefile']
              ORDER_ID  ...                                           geometry
            0    69356  ...  POLYGON ((380323.430 961539.720, 698840.025 24...
            [1 rows x 4 columns]
        """

        path_to_file = cd(self.DATA_DIR, self.PROJ_DIRNAME, self.PROJ_FILENAME)

        dgn_shp_prj_file = get_dgn_shp_prj(
            path_to_file=path_to_file, update=update, projcs_parser=parser, as_dict=as_dict,
            verbose=verbose)

        return dgn_shp_prj_file

    def dgn2shp(self, dat_name='Stations', confirmation_required=True, verbose=True, **kwargs):
        """
        Convert DGN files of OPAS data to shapefiles.

        :param dat_name: Name of the data; defaults to ``'Stations'``.
        :type dat_name: str | None
        :param confirmation_required: Whether to ask for confirmation to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the :func:`~src.utils.dgn.dgn2shp`
            function.

        **Examples**::

            >>> from src.preprocessor import CNM
            >>> opas = OPAS()
            >>> opas.dgn2shp()
            To convert .dgn files of "Stations" to shapefiles?
             [No]|Yes: yes
            Converting "ENG_ADMIN.RINM_OPAS_Stations.dgn" at "data\\OPAS\\Stations" ... Done.
        """

        if dat_name in [self.STN_DIRNAME]:
            dgn2shp_batch(
                dat_name=dat_name, file_paths=glob.glob(cd(self.DATA_DIR, dat_name, "*.dgn")),
                confirmation_required=confirmation_required, verbose=verbose,
                **kwargs)

        elif dat_name is None:
            if confirmed("To convert all available .dgn files of \"{}\" to shapefiles?"
                         "\n".format(self.ACRONYM), confirmation_required=confirmation_required):
                paths_to_dgn_files = glob.glob(cd(self.DATA_DIR, "*", "*.dgn"))

                dgn2shp_batch(
                    self.ACRONYM, paths_to_dgn_files, confirmation_required=False, verbose=verbose,
                    **kwargs)

    def read_stations_shp(self, update=False, verbose=False):
        """
        Read the shapefile of stations (converted from DGN data) from a local directory.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Shapefile data of stations converted from DGN files.
        :rtype: dict

        **Examples**::

            >>> from src.preprocessor import OPAS
            >>> opas = OPAS()
            >>> stn_shp = opas.read_stations_shp()
            >>> type(stn_shp)
            dict
            >>> list(stn_shp.keys())
            ['Annotation', 'MultiPatch', 'Point', 'Polygon', 'Polyline']
            >>> stn_shp['Annotation'].empty
            True
            >>> stn_shp['MultiPatch'].empty
            True
            >>> stn_shp['Point'].empty
            False
            >>> stn_shp['Polygon'].empty
            True
            >>> stn_shp['Polyline'].empty
            True
            >>> stn_shp['Point'].head()
                 Entity  ...                               geometry
            0     Point  ...   POINT Z (164858.964 39701.630 0.000)
            1     Point  ...  POINT Z (167561.310 797058.607 0.000)
            2     Point  ...  POINT Z (176248.346 827083.735 0.000)
            3     Point  ...   POINT Z (181803.496 32372.755 0.000)
            4     Point  ...  POINT Z (190012.468 206251.431 0.000)
            [5 rows x 30 columns]
        """

        path_to_dir = cd(self.DATA_DIR, self.STN_DIRNAME)
        dgn_filename = "ENG_ADMIN.RINM_OPAS_Stations.dgn"

        path_to_pickle = cd(path_to_dir, dgn_filename.replace(".dgn", ".pkl"))

        if os.path.isfile(path_to_pickle) and not update:
            stations_shp = load_data(path_to_pickle)

        else:
            if verbose:
                print("Parsing the shapefile of {}".format(self.STN_DIRNAME.lower()), end=" ... ")

            try:
                stations_shp = read_dgn_shapefile(path_to_dir, dgn_filename)

                print("Done.") if verbose else ""

                save_data(stations_shp, path_to_pickle, verbose=verbose)

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")
                stations_shp = None

        return stations_shp

    def import_stations_shp(self, update=False, confirmation_required=True, verbose=True, **kwargs):
        """
        Import shapefile data (converted from the DGN data) of stations into the project database.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param confirmation_required: Whether to ask for confirmation to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.import_data`_.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.preprocessor import OPAS
            >>> opas = OPAS()
            >>> opas.import_stations_shp(if_exists='replace')
            To import shapefile of stations into the table "OPAS"."Stations"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/opas_stations_shp_tbl.*
            :name: opas_stations_shp_tbl
            :align: center
            :width: 100%

            Snapshot of the "OPAS"."Stations" table.
        """

        dat_name = f"shapefile of {self.STN_DIRNAME.lower()}"
        tbl_name = f"\"{self.SCHEMA_NAME}\".\"{self.STN_TABLE_NAME}\""

        if confirmed(f"To import {dat_name} into the table {tbl_name}?\n", confirmation_required):
            stations_shp = self.read_stations_shp(update=update, verbose=verbose)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing the data", end=" ... ")

            try:
                shp_dat = stations_shp['Point']

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)

                    if shp_dat is not None:
                        shp_dat['geometry'] = shp_dat['geometry'].map(lambda x: x.wkt)

                self.db_instance.import_data(
                    data=shp_dat, schema_name=self.SCHEMA_NAME, table_name=self.STN_TABLE_NAME,
                    method=self.db_instance.psql_insert_copy, confirmation_required=False,
                    **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")

    def map_view(self, item_name='Stations', layer_name='Point', desc_col_name='Handle',
                 sample=True, marker_colour='darkred', update=False, verbose=True):
        """
        Make a map view of a given item.

        :param item_name: Name of the item; defaults to ``'Stations'``.
        :type item_name: str
        :param layer_name: Name of the layer; defaults to ``'Point'``.
        :type layer_name: str
        :param desc_col_name: Name of the column that describes markers; defaults to ``'Handle'``.
        :type desc_col_name: str
        :param sample: Whether to draw a sample or specify a sample size.
        :type sample: bool | int
        :param marker_colour: Colour of the markers; defaults to ``'darkred'``.
        :type marker_colour: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``True``.
        :type verbose: bool | int

        **Examples**::

            >>> from src.preprocessor import OPAS
            >>> opas = OPAS()
            >>> opas.map_view(desc_col_name='Handle', sample=100)

        .. raw:: html

            <iframe src="../_static/view_stations.html" marginwidth="0"
                marginheight="0" scrolling="no"
                style="width:100%; height:600px; border:0; overflow:hidden;">
            </iframe>

        .. only:: html

            .. figure:: ../_images/blank.png
                :name: blank_view_stations
                :align: center
                :width: 0%

                Examples of stations.

        .. only:: latex

            .. figure:: ../_images/opas_stations_view_demo.*
                :name: opas_stations_view_demo
                :align: center
                :width: 100%

                Eamples of stations.
        """

        dgn_shp_map_view(
            self, item_name=item_name, layer_name=layer_name, desc_col_name=desc_col_name,
            sample=sample, marker_colour=marker_colour, update=update, verbose=verbose)

    def load_stations_shp(self, **kwargs):
        """
        Load the shapefile data (converted from the DGN data) of stations from the project database.

        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.read_sql_query
            <https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.read_sql_query.html>`_.
        :return: DGN-converted shapefile data of stations.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import OPAS
            >>> opas = OPAS()
            >>> stn_shp = opas.load_stations_shp()
            >>> stn_shp.head()
                 Entity  ...                                         geometry
            0     Point  ...  POINT Z (164858.9644999998 39701.62969999947 0)
            1     Point  ...  POINT Z (167561.3097000001 797058.6073000003 0)
            2     Point  ...        POINT Z (176248.3465 827083.7353000008 0)
            3     Point  ...  POINT Z (181803.4960000003 32372.75510000065 0)
            4     Point  ...  POINT Z (190012.4675000003 206251.4306000005 0)
            [5 rows x 30 columns]
        """

        if self.db_instance is None:
            self.db_instance = TrackFixityDB()

        sql_query = f'SELECT * FROM "{self.SCHEMA_NAME}"."{self.STN_TABLE_NAME}"'

        stations_shp = self.db_instance.read_sql_query(sql_query=sql_query, **kwargs)

        if 'geometry' in stations_shp.columns:
            stations_shp.geometry = stations_shp.geometry.map(shapely.wkt.loads)

        return stations_shp
