"""
A class for preprocessing *point cloud data (PCD)*.
"""

import copy
import datetime
import gc
import glob
import itertools
import numbers
import os
import re
import shutil
import sys
import uuid
import webbrowser

import folium.plugins
import laspy
import natsort
import numpy as np
import pandas as pd
import pyproj
import shapely.wkt
import sqlalchemy
from pyhelpers._cache import _print_failure_msg
from pyhelpers.dirs import cd, cdd
from pyhelpers.geom import get_rectangle_centroid
from pyhelpers.ops import confirmed, get_obj_attr, gps_time_to_utc, swap_cols
from pyhelpers.store import load_data, save_data
from pyhelpers.text import find_similar_str, get_acronym
from sklearn.preprocessing import minmax_scale

from src.utils.dgn import (dgn2shp, dgn_shapefiles, fix_folium_float_image, read_dgn_shapefile,
                           remove_dgn_shapefiles)
from src.utils.general import (TrackFixityDB, add_sql_query_date_condition, cd_docs_source,
                               data_date_to_year_month, get_tile_xy)
from src.utils.geometry import flatten_geometry


class PCD:
    """
    *Point cloud data*.

    In this project, there are three types of point cloud data (PCD), including:

        - raw data (.LAZ/.LAS) from scanners
        - data in DGN format
        - data in KRDZ format
    """

    #: Data name.
    NAME: str = 'Point cloud data'
    #: Acronym for the data name.
    ACRONYM: str = get_acronym(NAME)
    #: Pathname of a local directory where the point cloud data is stored.
    DATA_DIR: str = os.path.relpath(cdd(ACRONYM))
    #: Short description of start and end mileages.
    MILEAGE: str = 'ECM7_67m_69m67ch_ECM8_10m_54m50ch'

    #: Filename of tiles data.
    TILES_FILENAME: str = "project.prj"
    #: Filename of the KRDZ schema file.
    KRDZ_SCHEMA_FILENAME: str = "KRDZ_schema"

    #: Name of the schema for storing the point cloud data.
    SCHEMA_NAME: str = copy.copy(ACRONYM)

    #: Name of the table storing the tile data.
    TILES_TABLE_NAME: str = 'Tiles'
    #: Name of the table storing the metadata of the tile data.
    TILES_META_TABLE_NAME: str = TILES_TABLE_NAME + '_Metadata'

    #: Name of the table storing the LAZ data.
    LAZ_TABLE_NAME: str = 'LAZ_OSGB_100x100'
    #: Name of the table storing the metadata of the LAZ data.
    LAZ_META_TABLE_NAME: str = LAZ_TABLE_NAME + '_Metadata'

    #: Name of the schema for storing the KRDZ data.
    KRDZ_SCHEMA_NAME: str = copy.copy(ACRONYM)
    #: Name of the table storing the KRDZ data.
    KRDZ_TABLE_NAME: str = 'KRDZ'
    #: Name of the table storing the metadata of KRDZ data.
    KRDZ_META_TABLE_NAME: str = KRDZ_TABLE_NAME + '_Metadata'
    #: Name of the table storing the classified KRDZ data.
    KRDZ_CL_TABLE_NAME: str = 'KRDZ_Classified'

    def __init__(self, elr='ECM8', db_instance=None):
        """
        :param elr: Engineer's Line Reference; defaults to ``'ECM8'``.
        :type elr: str
        :param db_instance: PostgreSQL database instance; defaults to ``None``.
        :type db_instance: TrackFixityDB | None

        :ivar str elr: Engineer's Line Reference.

        :ivar typing.Callable dgn_dir_: Pathname of the local directory for storing DGN files
            (for a given date).
        :ivar typing.Callable laz_dir_: Pathname of the local directory for storing LAZ files
            (for a given date).
        :ivar typing.Callable krdz_dir_: Pathname of the local directory for storing KRDZ files
            (for a given date).

        :ivar list data_dates: Dates of the available data.

        :ivar typing.Callable dgn_filename_: Filename of DGN data.
        :ivar typing.Callable dgn_shp_filename_: Filename of shapefile data
            converted from a DGN file.
        :ivar typing.Callable krdz_filename_: Filename of KRDZ data.

        :ivar pyhelpers.dbms.PostgreSQL db_instance: PostgreSQL database instance.

        :ivar typing.Callable laz_table_name_: Name of the table storing the LAZ data
            for a given date.
        :ivar typing.Callable laz_meta_table_name_: Name of the table storing the metadata
            of the LAZ data for a given date.
        :ivar typing.Callable dgn_shp_table_name_: Name of the table storing the DGN-converted
            shapefiles.

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> pcd.NAME
            'Point cloud data'
            >>> pcd.SCHEMA_NAME
            'PCD'

        .. figure:: ../_images/pcd_schema.*
            :name: pcd_schema
            :align: center
            :width: 100%

            Snapshot of the *PCD* schema.
        """

        self.elr = elr

        # Directories
        self.laz_dir_ = lambda pcd_date: cd(self.DATA_DIR, self.elr, "LAZ_OSGB_100x100", pcd_date)
        self.dgn_dir_ = lambda pcd_date: cd(self.DATA_DIR, self.elr, "DGN", pcd_date)
        self.krdz_dir_ = lambda pcd_date: cd(self.DATA_DIR, self.elr, "KRDZ", pcd_date)
        self.data_dates = [
            x for x in os.listdir(os.path.dirname(self.laz_dir_("")))
            if os.path.isdir(cd(os.path.dirname(self.laz_dir_("")), x))]

        # Filenames
        self.dgn_filename_ = lambda pcd_date: f"{self.elr}_10_66_{pcd_date}_20200625.dgn"
        self.dgn_shp_filename_ = lambda x, y: (
                self.dgn_filename_(x).replace(".", "_") + "_" + y + ".shp")
        self.krdz_filename_ = lambda x: f"{self.elr}_10_66_{x}_20200625.KRDZ"

        # Database
        self.db_instance = db_instance

        self.laz_table_name_ = lambda pcd_date: (
                self.LAZ_TABLE_NAME + ('_' + pcd_date if pcd_date else ''))
        self.laz_meta_table_name_ = lambda pcd_date: self.laz_table_name_(pcd_date) + '_Metadata'
        # Table names - DGN data
        self.dgn_shp_table_name_ = lambda lyr: f'DGN_Shapefile_{lyr}'

    def read_tiles_prj_by_date(self, pcd_date, update=False, verbose=False):
        """
        Read a PRJ file of tiles for a given date from a local directory.

        :param pcd_date: Date of the point cloud data in the format 'YYYY-MM-DD'.
        :type pcd_date: str
        :param update: Whether to reprocess the original data files; defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Metadata and information of each tile.
        :rtype: dict | None

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()

        **October 2019**::

            >>> project_prj = pcd.read_tiles_prj_by_date(pcd_date='201910')
            >>> type(project_prj)
            dict
            >>> list(project_prj.keys())
            ['Metadata', 'Projection']
            >>> project_prj['Metadata'].shape
            (12, 2)
            >>> project_prj['Projection'].shape
            (1933, 4)

        **April 2020**::

            >>> project_prj = pcd.read_tiles_prj_by_date(pcd_date='202004')
            >>> type(project_prj)
            dict
            >>> list(project_prj.keys())
            ['Metadata', 'Projection']
            >>> project_prj['Metadata'].shape
            (12, 2)
            >>> project_prj['Projection'].shape
            (2048, 4)
        """

        path_to_prj = cd(self.laz_dir_(pcd_date), self.TILES_FILENAME)

        path_to_pickle = path_to_prj.replace(".prj", ".pkl")

        if os.path.isfile(path_to_pickle) and not update:
            tiles_prj = load_data(path_to_pickle)

        else:
            if verbose:
                print(f"Parsing \"{self.TILES_FILENAME}\" of \"{pcd_date}\"", end=" ... ")

            try:
                with open(path_to_prj, "r") as f:
                    _ = f.readline()
                    f_ = f.read().rstrip('\n').split('\n\n')
                    meta_dat, tiles = f_[0].split('\n'), f_[1:]
                    f.close()

                # Metadata
                metadata = [x.split('=') for x in meta_dat]
                for i, x in enumerate(metadata):
                    try:
                        metadata[i] = [x[0], int(x[1])]
                    except ValueError:
                        pass
                # meta_dat_, bp = meta_dat_[:-1], meta_dat_[-1][0]

                prj_metadata = pd.DataFrame(metadata, columns=['Param', 'Value'])

                # Block tiles
                transformer = pyproj.Transformer.from_crs('EPSG:27700', 'EPSG:4326')

                laz_filenames, tile_xy, tile_lonlat = [], [], []
                for block_tile in tiles:
                    bt = block_tile.split('\n ')
                    laz_filenames.append(re.search('(?<=Block ).*', bt[0]).group(0))

                    tile_poly = [[float(x_) for x_ in x] for x in [x.split('  ') for x in bt[1:]]]
                    poly_xy = shapely.geometry.Polygon(tile_poly)
                    tile_xy.append(shapely.geometry.Polygon(poly_xy))

                    poly_lonlat = np.array(poly_xy.boundary.xy)
                    poly_ = np.array(transformer.transform(poly_lonlat[0], poly_lonlat[1]))
                    tile_lonlat.append(shapely.geometry.Polygon(poly_[[1, 0]].T))

                prj_data = pd.DataFrame(
                    {'LAZ_Filename': laz_filenames,
                     'Tile_Name': [x.replace('.laz', '') for x in laz_filenames],
                     'Tile_XY': tile_xy,
                     'Tile_LonLat': tile_lonlat})

                tiles_prj = {'Metadata': prj_metadata, 'Projection': prj_data}

                if verbose:
                    print("Done.")

                save_data(tiles_prj, path_to_pickle, verbose=verbose)

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")
                tiles_prj = None

        return tiles_prj

    def read_tiles_prj(self, update=False, verbose=False):
        """
        Read all PRJ files for tiles from a local directory.

        :param update: Whether to reprocess the original data files; defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Metadata and information of each tile as a dictionary.
        :rtype: dict

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> project_prj = pcd.read_tiles_prj()
            >>> type(project_prj)
            dict
            >>> list(project_prj.keys())
            ['Metadata', 'Projection']
            >>> project_prj['Metadata'].shape
            (24, 4)
            >>> project_prj['Projection'].shape
            (3981, 6)
        """

        path_to_pickle = cd(self.DATA_DIR, self.TILES_FILENAME.replace(".prj", ".pkl"))

        if os.path.isfile(path_to_pickle) and not update:
            tiles_prj = load_data(path_to_pickle)

        else:
            prj_metadata, prj_data = [], []

            for pcd_date in self.data_dates:
                tiles_prj_ = self.read_tiles_prj_by_date(pcd_date, update=update, verbose=verbose)

                prj_meta_, prj_data_ = tiles_prj_.values()

                pcd_date_ = datetime.datetime.strptime(pcd_date, '%Y%m').date()
                prj_meta_.insert(0, 'Year', pcd_date_.year)
                prj_meta_.insert(1, 'Month', pcd_date_.month)
                prj_data_.insert(0, 'Year', pcd_date_.year)
                prj_data_.insert(1, 'Month', pcd_date_.month)

                prj_metadata.append(prj_meta_)
                prj_data.append(prj_data_)

            tiles_prj = {'Metadata': pd.concat(prj_metadata, axis=0, ignore_index=True),
                         'Projection': pd.concat(prj_data, axis=0, ignore_index=True)}

            save_data(tiles_prj, path_to_pickle, verbose=verbose)

        return tiles_prj

    def map_view_tiles_by_date(self, pcd_date, tile_colour='#3186CC', update=False, verbose=True):
        """
        Make a map view of tiles of point cloud data for a given date.

        :param pcd_date: Date of the point cloud data in the format 'YYYY-MM-DD'.
        :type pcd_date: str
        :param tile_colour: Hex colour code for tiles on the map view; defaults to ``'#3186CC'``.
        :type tile_colour: str
        :param update: Whether to reprocess the original data files; defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``True``.
        :type verbose: bool | int

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()

        *October 2019*::

            >>> pcd.map_view_tiles_by_date(pcd_date='201910', tile_colour='#DB7B2B')

        .. raw:: html

            <iframe title="Tiles for the point cloud data (October 2019)."
                src="../_static/pcd_map_view_tiles_201910.html"
                marginwidth="0" marginheight="0" scrolling="no"
                style="width:100%; height:600px; border:0; overflow:hidden;">
            </iframe>

        .. only:: html

            .. figure:: ../_images/blank.png
                :name: blank_pcd_map_view_tiles_201910
                :align: center
                :width: 0%

                Tiles for the point cloud data (October 2019).

        .. only:: latex

            .. figure:: ../_images/pcd_map_view_tiles_201910.*
                :name: pcd_map_view_tiles_201910
                :align: center
                :width: 100%

                Tiles for the point cloud data (October 2019).

            .. figure:: ../_images/pcd_map_view_tiles_201910_zoomed_in.*
                :name: pcd_map_view_tiles_201910_zoomed_in
                :align: center
                :width: 100%

                Tiles (zoomed in) for the point cloud data (October 2019).

        *April 2020*::

            >>> pcd.map_view_tiles_by_date(pcd_date='202004', tile_colour='#3186CC')

        .. raw:: html

            <iframe title="Tiles for the point cloud data (April 2020)."
                src="../_static/pcd_map_view_tiles_202004.html"
                marginwidth="0" marginheight="0" scrolling="no"
                style="width:100%; height:600px; border:0; overflow:hidden;">
            </iframe>

        .. only:: html

            .. figure:: ../_images/blank.png
                :name: blank_pcd_map_view_tiles_202004
                :align: center
                :width: 0%

                Tiles for the point cloud data (April 2020).

        .. only:: latex

            .. figure:: ../_images/pcd_map_view_tiles_202004.*
                :name: pcd_map_view_tiles_202004
                :align: center
                :width: 100%

                Tiles for the point cloud data (April 2020).

            .. figure:: ../_images/pcd_map_view_tiles_202004_zoomed_in.*
                :name: pcd_map_view_tiles_202004_zoomed_in
                :align: center
                :width: 100%

                Tiles (zoomed in) for the point cloud data (April 2020).
        """

        m_filename = f"pcd_map_view_tiles_{pcd_date}.html"
        path_to_m = cd(self.laz_dir_(pcd_date), m_filename)

        if not os.path.isfile(path_to_m) or update:

            tiles_prj = self.read_tiles_prj_by_date(pcd_date=pcd_date, update=update, verbose=verbose)
            block_tiles = tiles_prj['Projection']
            tiles_poly = shapely.geometry.MultiPolygon(list(block_tiles['Tile_LonLat']))

            m_loc = get_rectangle_centroid(tiles_poly.geoms)  # np.array(tiles_poly.centroid.coords)[0]
            # Note that `m_loc` presents as [Longitude, Latitude]
            m = folium.Map(location=[m_loc[1], m_loc[0]], zoom_start=10, control_scale=True)

            folium.plugins.MiniMap(zoom_level_offset=-6).add_to(m)

            polygons = block_tiles['Tile_LonLat'].map(
                lambda poly: list(swap_cols(np.array(poly.boundary.xy).T, 0, 1, as_list=True)[:-1]))
            tile_xys = np.array(
                block_tiles['Tile_XY'].map(lambda poly: poly.exterior.coords[0]).to_list())
            laz_filenames = block_tiles['Tile_Name']

            for polygon, label, laz_fn in zip(polygons, tile_xys.astype(int), laz_filenames):
                folium.Polygon(
                    locations=polygon,
                    popup=folium.Popup(
                        'Tile: {}<br>Laz filename: {}'.format(tuple(label), laz_fn), max_width=500),
                    color=tile_colour,
                    fill=True,
                    fill_color=tile_colour,
                    fillOpacity=0.5,
                ).add_to(m)

            folium.plugins.FloatImage(
                image=cdd(self.DATA_DIR, "north_arrow.png"),
                bottom=8, left=1,  # Or, bottom=1, left=8
                width='22px').add_to(m)

            m.save(path_to_m)

            fix_folium_float_image(path_to_m)

            shutil.copyfile(path_to_m, cd_docs_source("_static", m_filename))

        webbrowser.open(path_to_m)

    def map_view_tiles(self, tile_colours=None, update=False, verbose=True):
        """
        Make a map view of tiles of all available point cloud data.

        :param tile_colours: List of hex colour codes for tiles on the map view;
            defaults to ``None``.
        :type tile_colours: list | None
        :param update: Whether to reprocess the original data files; defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> pcd.map_view_tiles()

        .. raw:: html

            <iframe title="Tiles for the point cloud data."
                src="../_static/pcd_map_view_tiles.html"
                marginwidth="0" marginheight="0" scrolling="no"
                style="width:100%; height:600px; border:0; overflow:hidden;">
            </iframe>

        .. only:: html

            .. figure:: ../_images/blank.png
                :name: blank_pcd_map_view_tiles
                :align: center
                :width: 0%

                Tiles for the point cloud data.

        .. only:: latex

            .. figure:: ../_images/pcd_map_view_tiles.*
                :name: pcd_map_view_tiles
                :align: center
                :width: 100%

                Tiles for the point cloud data.

            .. figure:: ../_images/pcd_map_view_tiles_zoomed_in.*
                :name: pcd_map_view_tiles_zoomed_in
                :align: center
                :width: 100%

                Tiles (zoomed in) for the point cloud data.
        """

        m_filename = "pcd_map_view_tiles.html"
        path_to_m = cd(self.laz_dir_(''), m_filename)

        tiles_prj = self.read_tiles_prj(update=update, verbose=verbose)

        block_tiles = tiles_prj['Projection']
        tiles_poly = shapely.geometry.MultiPolygon(list(block_tiles.Tile_LonLat))

        polygons = block_tiles.Tile_LonLat.map(
            lambda poly: list(swap_cols(np.array(poly.boundary.xy).T, 0, 1, as_list=True)[:-1]))
        tile_xys = block_tiles.Tile_XY.map(lambda poly: tuple(int(x) for x in poly.exterior.coords[0]))
        laz_filenames = block_tiles.Tile_Name

        m_loc = get_rectangle_centroid(tiles_poly.geoms)  # [Longitude, Latitude]
        m = folium.Map(location=[m_loc[1], m_loc[0]], tiles=None, zoom_start=10, control_scale=True)
        folium.TileLayer('OpenStreetMap', name='Tiles for the point cloud data').add_to(m)

        folium.plugins.MiniMap(zoom_level_offset=-6).add_to(m)

        if tile_colours is None:
            tile_colours = ['#DB7B2B', '#3186CC']

        if not os.path.isfile(path_to_m) or update:

            for pcd_date, tile_colour in zip(self.data_dates, tile_colours):

                i = block_tiles[block_tiles.Year == int(pcd_date[:4])].index

                # Create a FeatureGroup
                pcd_date_lyr = folium.FeatureGroup(name=pcd_date[:4] + '-' + pcd_date[4:], control=True)

                for polygon, label, laz_fn in zip(polygons[i], tile_xys[i], laz_filenames[i]):
                    folium.Polygon(
                        locations=polygon,
                        popup=folium.Popup(
                            'Tile: {}<br>Laz filename: {}'.format(tuple(label), laz_fn), max_width=500),
                        color=tile_colour,
                        fill=True,
                        fill_color=tile_colour,
                        fillOpacity=0.5,
                    ).add_to(pcd_date_lyr)

                m.add_child(pcd_date_lyr)

            # folium.LayerControl(collapsed=False).add_to(m)

            folium.plugins.FloatImage(
                image=cdd(self.DATA_DIR, "north_arrow.png"),
                bottom=8, left=1,  # bottom=1, left=8
                width='22px').add_to(m)

            m.save(path_to_m)

            fix_folium_float_image(path_to_m)

            shutil.copyfile(path_to_m, cd_docs_source("_static", m_filename))

        webbrowser.open(path_to_m)

    def import_tiles(self, update=False, confirmation_required=True, verbose=True, **kwargs):
        """
        Import information about tiles into the project database.

        :param update: Whether to reprocess the original data files; defaults to ``False``.
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

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> pcd.import_tiles()
            To import data of tiles into the schema "PCD"?
             [No]|Yes: yes
            Importing the metadata ... Done.
            Importing information about coordinates of the tiles ... Done.

        .. figure:: ../_images/pcd_tiles_tbl.*
            :name: pcd_tiles_tbl
            :align: center
            :width: 100%

            Snapshot of the "PCD"."Tiles" table.

        .. figure:: ../_images/pcd_tiles_metadata_tbl.*
            :name: pcd_tiles_metadata_tbl
            :align: center
            :width: 100%

            Snapshot of the "PCD"."Tiles_Metadata" table.
        """

        if confirmed("To import data of tiles into the schema \"{}\"?\n".format(self.SCHEMA_NAME),
                     confirmation_required=confirmation_required):

            tiles_prj = self.read_tiles_prj(update=update, verbose=verbose)

            tiles_metadata, block_tiles = tiles_prj.values()

            poly_cols = ['Tile_XY', 'Tile_LonLat']
            block_tiles[poly_cols] = block_tiles[poly_cols].map(lambda x: x.wkt)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing the metadata", end=" ... ")

            try:
                self.db_instance.import_data(
                    data=tiles_metadata, schema_name=self.SCHEMA_NAME,
                    table_name=self.TILES_META_TABLE_NAME,
                    method=self.db_instance.psql_insert_copy, confirmation_required=False,
                    **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e)

            if verbose:
                print("Importing information about coordinates of the tiles", end=" ... ")

            try:
                self.db_instance.import_data(
                    data=block_tiles, schema_name=self.SCHEMA_NAME, table_name=self.TILES_TABLE_NAME,
                    method=self.db_instance.psql_insert_copy, confirmation_required=False,
                    **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")

    def load_tiles(self, pcd_date=None):
        """
        Load data of the tiles for point cloud data.

        :param pcd_date: Date of the point cloud data in the format 'YYYY-MM-DD';
            defaults to ``None``.
        :type pcd_date: str | int | None
        :return: Data of tiles.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> tiles_dat = pcd.load_tiles(pcd_date='201910')
            >>> tiles_dat.head()
                  Year  ...                                        Tile_LonLat
            0     2019  ...  POLYGON ((-2.960910975517333 55.95617100494877...
            1     2019  ...  POLYGON ((-2.960888715229412 55.95527265723062...
            2     2019  ...  POLYGON ((-2.012726922542862 55.77482976892926...
            3     2019  ...  POLYGON ((-2.009538521884446 55.77393156353337...
            4     2019  ...  POLYGON ((-2.01113257284158 55.77393142614289,...
            [5 rows x 6 columns]
            >>> tiles_dat = pcd.load_tiles(pcd_date='202004')
            >>> tiles_dat.head()
                  Year  ...                                        Tile_LonLat
            0     2020  ...  POLYGON ((-2.962512280825055 55.95615850596208...
            1     2020  ...  POLYGON ((-2.960888715229412 55.95527265723062...
            2     2020  ...  POLYGON ((-2.960910975517333 55.95617100494877...
            3     2020  ...  POLYGON ((-2.960933237072112 55.95706935252924...
            4     2020  ...  POLYGON ((-2.959287445726476 55.95528513499585...
            [5 rows x 6 columns]
            >>> tiles_dat = pcd.load_tiles()
            >>> tiles_dat.shape
            (3981, 6)
        """

        if self.db_instance is None:
            self.db_instance = TrackFixityDB()

        query = f'SELECT * FROM "{self.SCHEMA_NAME}"."{self.TILES_TABLE_NAME}"'

        if pcd_date is not None:
            year, month = data_date_to_year_month(data_date=pcd_date)
            query += f' WHERE "Year"={year} AND "Month"={month}'

        pcd_tiles = self.db_instance.read_sql_query(sql_query=query, method='tempfile')

        return pcd_tiles

    # == LAZ data ==================================================================================

    @staticmethod
    def _parse_laz_header_data(x):
        """
        Parse the header data of a LAZ (.laz) file.

        :param x: One element of the LAZ (.laz) file.
        :type x: typing.Any
        :return: Parsed data.
        :rtype: typing.Any

        .. warning::

            This method may need modifications.
        """

        if isinstance(x, bytes):
            y = x.decode()

        elif isinstance(x, (laspy.header.GlobalEncoding, laspy.point.format.PointFormat)):
            # This designates properties about the file,
            # such as how to interpret GPS time and CRS information.
            attrs = [x_ for x_ in dir(x) if not x_.startswith('_') and not callable(getattr(x, x_))]
            x_ = [
                [x.__getattribute__(a).name, x.__getattribute__(a).value]
                if isinstance(x.__getattribute__(a), laspy.header.GpsTimeType)
                else x.__getattribute__(a)
                for a in attrs]
            y = dict(zip(attrs, x_))

        elif isinstance(x, uuid.UUID):
            y = str(x)

        elif x is None:
            y = ''

        elif not isinstance(x, (str, list, numbers.Number, datetime.date)):
            y = list(x)

        else:
            y = x

        return y

    def parse_laz(self, path_to_laz, header_to_dataframe=True, points_to_dataframe=True, **kwargs):
        # noinspection PyShadowingNames
        """
        Transform LAS/LAZ data into a dataframe.

        :param path_to_laz: Pathname of a LAZ/LAS file.
        :type path_to_laz: str
        :param header_to_dataframe: Whether to convert the header data to a dataframe;
            defaults to ``True``.
        :type header_to_dataframe: bool
        :param points_to_dataframe: Whether to convert the points data to a dataframe;
            defaults to ``True``.
        :type points_to_dataframe: bool
        :param kwargs: [Optional] additional parameters for `pylas.read`_.
        :return: LAS/LAZ data, including header and points.
        :rtype: tuple

        .. _`pylas.read`: https://pylas.readthedocs.io/en/latest/api/index.html#pylas.read

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> from pyhelpers.dirs import cd
            >>> pcd = PCD()
            >>> tile_x, tile_y = (340500, 674200)
            >>> path_to_laz = cd(pcd.laz_dir_('201910'), f"Tile_X+0000{tile_x}_Y+0000{tile_y}.laz")
            >>> header_data, laz_points_data = pcd.parse_laz(path_to_laz)
            >>> header_data
                                            DEFAULT_POINT_FORMAT  ... z_scale
            0  {'dimension_names': <generator object PointFor...  ...   0.001
            [1 rows x 39 columns]
            >>> laz_points_data.head()
                               X           Y       Z  ...  red  green  blue
            0         340600.000  674239.695  41.676  ...  255    255   255
            1         340600.000  674240.305  40.457  ...  255    255   255
            2         340600.000  674239.609  41.880  ...  255    255   255
            3         340600.000  674238.253  41.249  ...  255    255   255
            4         340600.000  674238.152  39.933  ...  255    255   255
            [5 rows x 13 columns]
            >>> header_data, laz_points_data = pcd.parse_laz(
            ...     path_to_laz, header_to_dataframe=False, points_to_dataframe=False)
            >>> type(header_data)
            dict
            >>> list(header_data.keys())[:5]
            ['DEFAULT_POINT_FORMAT',
             'DEFAULT_VERSION',
             'are_points_compressed',
             'creation_date',
             'evlrs']
            >>> type(laz_points_data)
            numpy.ndarray
            >>> laz_points_data.shape
            (54007138, 13)
        """

        # kwargs ={'laz_backend': cd(self.DataDirPath, "laszip.exe")}
        laz_file = laspy.read(path_to_laz, **kwargs)

        # Header
        header_data = get_obj_attr(laz_file.header, col_names=[1, 0], as_dataframe=True).set_index(1)

        if header_to_dataframe:
            header_data = get_obj_attr(
                laz_file.header, col_names=[1, 0], as_dataframe=True).set_index(1).T
            # noinspection PyTypeChecker
            header_data.rename_axis(None, axis=1, inplace=True)
            header_data = header_data.map(self._parse_laz_header_data)

            scales, offsets = header_data.scales[0], header_data.offsets[0]

        else:
            # noinspection PyTypeChecker
            header_data.rename_axis(None, axis=0, inplace=True)
            header_data = header_data[0].map(self._parse_laz_header_data).to_dict()
            scales, offsets = header_data['scales'], header_data['offsets']

        # Points
        laz_points_data = laz_file.points

        if points_to_dataframe:
            laz_points_data = pd.DataFrame(laz_points_data.array)

            # Points - XYZ
            laz_points_data[['X', 'Y', 'Z']] = laz_points_data[['X', 'Y', 'Z']] * scales + offsets

            # Points - GPS time
            laz_points_data.gps_time = laz_points_data.gps_time.map(gps_time_to_utc)

        else:
            laz_points_data = np.array(laz_file.points.array.tolist())
            laz_points_data[:, :3] = laz_points_data[:, :3] * scales + offsets

        return header_data, laz_points_data

    def _import_laz(self, paths_to_laz_files, laz_table_name, subset, only_metadata, incl_metadata,
                    laz_metadata_table_name, if_exists, verbose, **kwargs):
        total_no = len(paths_to_laz_files)

        if subset:
            assert max(subset) < total_no
            paths_to_laz_files_ = [paths_to_laz_files[i] for i in subset]
        else:
            paths_to_laz_files_ = copy.copy(paths_to_laz_files)

        if self.db_instance is None:
            self.db_instance = TrackFixityDB()

        if verbose:
            print("Importing the data ... ")

        for path_to_laz_file in paths_to_laz_files_:
            current_file_no = paths_to_laz_files_.index(path_to_laz_file) + 1
            filename = os.path.basename(path_to_laz_file)

            tile_x, tile_y = get_tile_xy(tile_xy=filename)

            if verbose:
                print(f"\t{current_file_no}/{total_no}: \"{filename}\"", end=" ... ")

            try:
                las_metadata, las_dat = self.parse_laz(path_to_laz_file)

                if not only_metadata:
                    # Tile_X and Tile_Y
                    las_dat['tile_X'], las_dat['tile_Y'] = tile_x, tile_y

                    las_data = las_dat.set_index(['gps_time', 'tile_X', 'tile_Y'])

                    las_data_size = sys.getsizeof(las_data) / (1024 * 1024)
                    chunk_no = int(np.round(las_data_size / 350))
                    # chunk_size = (len(las_data) // chunk_no) if las_data_size > 1 else None
                    if chunk_no > 1:
                        las_data = np.array_split(las_data, chunk_no)
                        gc.collect()

                    self.db_instance.import_data(
                        data=las_data, schema_name=self.SCHEMA_NAME, table_name=laz_table_name,
                        index=True, if_exists=if_exists, method=self.db_instance.psql_insert_copy,
                        confirmation_required=False, **kwargs)

                    del las_data
                    gc.collect()

                if incl_metadata:
                    las_metadata['tile_X'], las_metadata['tile_Y'] = tile_x, tile_y

                    self.db_instance.import_data(
                        data=las_metadata.set_index(['tile_X', 'tile_Y']),
                        schema_name=self.SCHEMA_NAME, table_name=laz_metadata_table_name,
                        index=True, if_exists=if_exists, method=self.db_instance.psql_insert_copy,
                        confirmation_required=False,
                        **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e)
                break

    def import_laz_by_date(self, pcd_date, subset=None, incl_metadata=True, only_metadata=False,
                           if_exists='fail', confirmation_required=True, verbose=True, **kwargs):
        # noinspection PyUnresolvedReferences
        """
        Import LAZ data for a given date into the project database.

        :param pcd_date: Date of the point cloud data in the format 'YYYY-MM-DD'.
        :type pcd_date: str
        :param incl_metadata: Whether to import the metadata along with the data;
            defaults to ``True``.
        :type incl_metadata: bool
        :param only_metadata: Whether to import only the metadata and not the data itself;
            defaults to ``False``.
        :type only_metadata: bool
        :param subset: List of indexes indicating specific local file paths to import;
            defaults to ``None``.
        :type subset: list | tuple | numpy.ndarray | range | None
        :param if_exists: Action to take if the targeted table already exists
            (see `pandas.DataFrame.to_sql`_); defaults to ``'fail'``.
        :type if_exists: str
        :param confirmation_required: Whether to ask for confirmation before proceeding;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.import_data`_.

        .. _`pandas.DataFrame.to_sql`:
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html
        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Test (201910)**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> pcd.import_laz_by_date(pcd_date='201910', if_exists='append')
            To import LAZ data (October 2019) into the table "PCD"."LAZ_OSGB_100x100_201910"?
             [No]|Yes: yes
            Importing the data ...
                1/1933: "Tile_X+0000374700_Y+0000673800.laz" ... Done.
                ...
                1933/1933: "Tile_X+0000399700_Y+0000654500.laz" ... Done.

        .. figure:: ../_images/pcd_laz_tbl_201910.*
            :name: pcd_laz_tbl_201910
            :align: center
            :width: 100%

            Snapshot of the "PCD"."LAZ_OSGB_100x100_201910" table.

        **Test (202004)**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> pcd.import_laz_by_date(pcd_date='202004', if_exists='append')
            To import LAZ data (April 2020) into the table "PCD"."LAZ_OSGB_100x100_202004"?
             [No]|Yes: yes
            Importing the data ...
                1/2048: "Tile_X+0000340000_Y+0000674100.laz" ... Done.
                ...
                2048/2048: "Tile_X+0000399700_Y+0000654500.laz" ... Done.

        .. figure:: ../_images/pcd_laz_tbl_202004.*
            :name: pcd_laz_tbl_202004
            :align: center
            :width: 100%

            Snapshot of the "PCD"."LAZ_OSGB_100x100_202004" table.

        **Create an index**::

            # SET max_parallel_maintenance_workers TO 10;
            SET max_parallel_workers TO 10;
            SET maintenance_work_mem TO '10 GB';
            SET checkpoint_timeout TO '180min';
            SET max_wal_size TO '100GB';
            SET min_wal_size TO '80MB';

            ALTER TABLE "PCD"."LAZ_OSGB_100x100_202004" SET (parallel_workers = 10);

            CREATE INDEX tiles_xy_202004 ON "PCD"."LAZ_OSGB_100x100_202004" ("tile_X", "tile_Y");

            SET enable_seqscan TO OFF;

        **Debugging (deprecated)**::

            >>> # Check if there are missing files that weren't imported into the database
            >>> from src.preprocessor import PCD
            >>> from src.utils import TrackFixityDB
            >>> from pyhelpers.dirs import cd
            >>> import glob
            >>> import natsort
            >>> pcd = PCD()
            >>> dat_date = '202004'
            >>> db_instance = TrackFixityDB()
            >>> with db_instance.engine.connect() as conn:
            ...     query = (f"SELECT table_name FROM information_schema.tables "
            ...              f"WHERE table_schema='LAZ_OSGB_100x100_{dat_date}' "
            ...              f"AND table_type='BASE TABLE';")
            ...     res = conn.execute(sqlalchemy.text(query))
            >>> temp_list = res.fetchall()
            >>> table_list = [tn[0] for tn in temp_list if tn[0].startswith("Tile_X")]
            >>> laz_dat_dir = cd(pcd.laz_dir_(dat_date))
            >>> file_list = natsort.natsorted(glob.glob1(cd(laz_dat_dir), "*.laz"))
            >>> file_list = [fn.replace(".laz", "") for fn in file_list]
            >>> # A list of paths to the missing files
            >>> missing_files = [cd(laz_dat_dir, x) for x in list(set(file_list) - set(table_list))]
            >>> missing_files
            []

        **PostgreSQL (PL/pgSQL) query for moving and renaming tables**:

        .. code-block:: plpgsql

            DO $$
            DECLARE
                tbl_name TEXT;
            BEGIN
                SET search_path='PointCloudData202004';
                FOR tbl_name IN
                    SELECT table_name FROM information_schema.tables
                    WHERE table_name like '%_202004_metadata'
                        AND table_schema='PointCloudData202004'
                        AND table_type='BASE TABLE'
                    ORDER BY table_name
                LOOP
                    EXECUTE FORMAT(
                        'ALTER TABLE "%s" SET SCHEMA "PointCloudMetadata202004";', tbl_name);
                END LOOP;
            END; $$;

            DO $$
            DECLARE
                tbl_name TEXT;
            BEGIN
                SET search_path='PointCloudMetadata202004';
                FOR tbl_name IN
                    SELECT table_name FROM information_schema.tables
                    WHERE table_name like '%_202004_metadata'
                        AND table_schema='PointCloudMetadata202004'
                        AND table_type='BASE TABLE'
                    ORDER BY table_name
                LOOP
                    EXECUTE FORMAT(
                        'ALTER TABLE "%s" RENAME TO "%s";', tbl_name, SUBSTRING(tbl_name, 1, 30));
                END LOOP;
            END; $$;


            DO $$
            DECLARE
                tbl_name TEXT;
            BEGIN
                SET search_path='PointCloudData202004';
                FOR tbl_name IN
                    SELECT table_name FROM information_schema.tables
                    WHERE table_name like '%_202004'
                        AND table_schema='PointCloudData202004'
                        AND table_type='BASE TABLE'
                    ORDER BY table_name
                LOOP
                    EXECUTE FORMAT(
                        'ALTER TABLE "%s" RENAME TO "%s";', tbl_name, SUBSTRING(tbl_name, 1, 30));
                END LOOP;
            END; $$;
        """

        assert pcd_date in self.data_dates, f"`pcd_date` must be one of {self.data_dates}."

        laz_table_name = self.laz_table_name_(pcd_date)
        laz_metadata_table_name = self.laz_meta_table_name_(pcd_date)

        data_date_ = datetime.datetime.strptime(pcd_date, '%Y%m').strftime('%B %Y')

        if not only_metadata:
            tbl_name = f'"{self.SCHEMA_NAME}"."{laz_table_name}"'
            dat_name = "data"
        else:
            tbl_name = f'"{self.SCHEMA_NAME}"."{laz_metadata_table_name}"'
            dat_name = "metadata"

        if confirmed(f"To import LAZ {dat_name} ({data_date_}) into the table {tbl_name}?\n",
                     confirmation_required=confirmation_required):
            laz_data_dir = self.laz_dir_(pcd_date)
            paths_to_laz_files = natsort.natsorted(glob.glob(cd(laz_data_dir, "*.laz")))

            self._import_laz(
                paths_to_laz_files, subset=subset, only_metadata=only_metadata,
                incl_metadata=incl_metadata, laz_table_name=laz_table_name,
                laz_metadata_table_name=laz_metadata_table_name, if_exists=if_exists,
                verbose=verbose, **kwargs)

    def import_laz(self, subset=None, incl_metadata=True, only_metadata=False, if_exists='append',
                   confirmation_required=True, verbose=True, **kwargs):
        """
        Import LAZ data into the project database.

        :param subset: List of indices indicating specific local file paths to import;
            defaults to ``None``.
        :type subset: list | tuple | numpy.ndarray | range | None
        :param incl_metadata: Whether to import the metadata along with the data;
            defaults to ``True``.
        :type incl_metadata: bool
        :param only_metadata: Whether to import only the metadata and not the data itself;
            defaults to ``False``.
        :type only_metadata: bool
        :param if_exists: Action to take if the targeted table already exists
            (see `pandas.DataFrame.to_sql`_); defaults to ``'append'``.
        :type if_exists: str
        :param confirmation_required: Whether to ask for confirmation before proceeding;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.import_data`_.

        .. _`pandas.DataFrame.to_sql`:
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html
        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Test (201910)**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> pcd.import_laz(if_exists='append')
            To import LAZ data into the schema "PCD"?
             [No]|Yes: yes
            Importing the data into "PCD"."LAZ_OSGB_100x100" ...
                1/3981: "Tile_X+0000340100_Y+0000674000.laz" ... Done.
                ...
                3981/3981: "Tile_X+0000399700_Y+0000654500.laz" ... Done.

        .. figure:: ../_images/pcd_laz_metadata_tbl.*
            :name: pcd_laz_metadata_tbl
            :align: center
            :width: 100%

            Snapshot of the "PCD"."LAZ_OSGB_100x100_Metadata" table.
        """

        if not only_metadata:
            tbl_name = f'"{self.SCHEMA_NAME}"."{self.LAZ_TABLE_NAME}"'
            dat_name = "data"
        else:
            tbl_name = f'"{self.SCHEMA_NAME}"."{self.LAZ_META_TABLE_NAME}"'
            dat_name = "metadata"

        if confirmed(f"To import LAZ {dat_name} into {tbl_name}?\n", confirmation_required):
            laz_data_dirs = []
            for pcd_date in self.data_dates:
                laz_data_dir = self.laz_dir_(pcd_date)
                laz_file_path = natsort.natsorted(glob.glob(cd(laz_data_dir, "*.laz")))
                laz_data_dirs.append(laz_file_path)
            paths_to_laz_files = list(itertools.chain.from_iterable(laz_data_dirs))

            self._import_laz(
                paths_to_laz_files, subset=subset, only_metadata=only_metadata,
                incl_metadata=incl_metadata, laz_table_name=self.LAZ_TABLE_NAME,
                laz_metadata_table_name=self.LAZ_META_TABLE_NAME, if_exists=if_exists,
                verbose=verbose, **kwargs)

    def check_pcd_dates(self, pcd_dates, len_req=None):
        """
        Check the dates provided as input for point cloud data.

        :param pcd_dates: Dates of the point cloud data.
        :type pcd_dates: int | str | list | tuple | None
        :param len_req: Length required for the range of the ``pcd_dates``; defaults to ``None``.
        :type len_req: int | None
        :return: List of validated data dates.
        :rtype: list

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> pcd.check_pcd_dates('201910')
            ['201910']
            >>> pcd.check_pcd_dates(['201910', 202004])
            ['201910', '202004']
        """

        if pcd_dates is None:
            pcd_dates_ = self.data_dates[-2:]
        elif isinstance(pcd_dates, (str, int)):
            pcd_dates_ = [str(pcd_dates)]
        else:
            assert isinstance(pcd_dates, (list, tuple))
            if len_req:
                assert len(pcd_dates) == len_req
            pcd_dates_ = [str(x) for x in pcd_dates]

        return pcd_dates_

    def load_laz(self, tile_xy, pcd_dates=None, greyscale=True, gs_coef=1.2, limit=None, **kwargs):
        """
        Load LAZ data within a given tile and dates of the point cloud data
        from the project database.

        :param tile_xy: Easting (X) and northing (Y) coordinates of the tile.
        :type tile_xy: tuple | list
        :param pcd_dates: Date(s) of the point cloud data; defaults to ``None``.
        :type pcd_dates: str | list | None
        :param greyscale: Whether to transform the colour data to greyscale; defaults to ``True``.
        :type greyscale: bool
        :param gs_coef: Coefficient for adjusting intensity (only when ``greyscale=True``);
            defaults to ``1.2``.
        :type gs_coef: float
        :param limit: Limit on the number of rows to query from the database; defaults to ``None``.
        :type limit: int
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.read_sql_query`_.
        :return: Dictionary containing XYZ and RGB data.
        :rtype: dict

        .. _`pyhelpers.dbms.PostgreSQL.read_sql_query`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.read_sql_query.html

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> xyz_rgb_dat = pcd.load_laz(tile_xy=(340500, 674200), limit=10000)
            >>> type(xyz_rgb_dat)
            dict
            >>> list(xyz_rgb_dat.keys())
            ['201910', '202004']
            >>> type(xyz_rgb_dat['201910'][0])
            numpy.ndarray
            >>> xyz_rgb_dat['201910'][0].shape
            (10000, 3)
            >>> type(xyz_rgb_dat['202004'][0])
            numpy.ndarray
            >>> xyz_rgb_dat['202004'][0].shape
            (10000, 3)
        """

        if self.db_instance is None:
            self.db_instance = TrackFixityDB()

        with self.db_instance.engine.connect() as connection:
            connection.execute(sqlalchemy.text('SET enable_seqscan TO OFF;'))
        # pcd.db_instance.engine.dialect.server_side_cursors = True
        # pcd.db_instance.engine.execution_options(stream_results=True)

        col_names = '"X", "Y", "Z", "red", "green", "blue", "intensity"'

        xyz_rgb_data = {}

        tile_x, tile_y = tile_xy

        for pcd_date in self.check_pcd_dates(pcd_dates):
            try:
                tbl_name = f'"{self.SCHEMA_NAME}"."{self.laz_table_name_(pcd_date)}"'

                query = f'SELECT {col_names} FROM {tbl_name} ' \
                        f'WHERE "tile_X"={tile_x} AND "tile_Y"={tile_y}'
                if limit:
                    query += f' LIMIT {limit}'

                xyz_rgb_tbl = self.db_instance.read_sql_query(
                    sql_query=query, method='tempfile', **kwargs)

            except Exception as e:
                _print_failure_msg(e)

                # Read point cloud data (from the local data directory)
                laz_filename = f"Tile_X+0000{tile_x}_Y+0000{tile_y}.laz"
                path_to_laz = cd(self.laz_dir_(pcd_date), laz_filename)

                _, xyz_rgb_tbl = self.parse_laz(path_to_laz)

                col_names_ = [eval(col_name) for col_name in col_names.split(', ')]
                xyz_rgb_tbl = xyz_rgb_tbl[col_names_]

            xyz_rgb = xyz_rgb_tbl.to_numpy()  # XYZ, RGB and intensity

            xyz = xyz_rgb[:, :3]
            rgb = xyz_rgb[:, 3:6]
            rgb.dtype = np.float64

            if greyscale:
                # Given the 'intensity', generate coefficients for the RGB data
                coef = minmax_scale(xyz_rgb_tbl.intensity).reshape((len(xyz_rgb_tbl.intensity), 1))
                rgb *= coef * gs_coef
                rgb[np.where(rgb > 255)] = 255

            xyz_rgb_data.update({pcd_date: [xyz, rgb]})

        return xyz_rgb_data

    # == DGN data ==================================================================================

    def dgn2shp(self, pcd_date=None, confirmation_required=True, verbose=True, **kwargs):
        """
        Convert DGN data to shapefiles for a specific date of the point cloud data.

        :param pcd_date: Date of the point cloud data, e.g., ``'201910'``, ``'202004'``.
            When ``pcd_date=None`` (default), the function converts all available
            "ECM8_10_66_*_20200625" DGN data.
        :type pcd_date: str | None
        :param confirmation_required: Whether to ask for confirmation to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the function
            :func:`~src.utils.dgn.dgn2shp`.

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> pcd.dgn2shp()
            To convert the DGN file "ECM8_10_66_201910_20200625.dgn" to shapefiles?
             [No]|Yes: yes
            Converting "ECM8_10_66_201910_20200625.dgn" at "data\\PCD\\ECM8\\DGN\\201910" ... Done.
            To convert the DGN file "ECM8_10_66_202004_20200625.dgn" to shapefiles?
             [No]|Yes: yes
            Converting "ECM8_10_66_202004_20200625.dgn" at "data\\PCD\\ECM8\\DGN\\202004" ... Done.
        """

        if pcd_date:
            path_to_dgn_files = [cd(self.dgn_dir_(pcd_date), self.dgn_filename_(pcd_date))]

        else:
            temp_lst = [glob.glob(cd(self.dgn_dir_(x), "*.dgn")) for x in self.data_dates]
            path_to_dgn_files = list(itertools.chain.from_iterable(temp_lst))

        for path_to_dgn in path_to_dgn_files:
            dgn_dat_dir, dgn_filename = os.path.split(path_to_dgn)

            if confirmed(f"To convert the DGN file \"{dgn_filename}\" to shapefiles?\n",
                         confirmation_required=confirmation_required):
                shp_filenames = [
                    dgn_filename.replace(".", "_") + "_" + x + ".shp" for x in dgn_shapefiles()]

                if all(not os.path.isfile(cd(dgn_dat_dir, f)) for f in shp_filenames):
                    dgn2shp(path_to_dgn, verbose=verbose, **kwargs)

                else:
                    if confirmed(f"Renew the existing .shp files for \"{dgn_filename}\"?\n",
                                 confirmation_required=confirmation_required):
                        remove_dgn_shapefiles(path_to_dgn)

                        dgn2shp(path_to_dgn, verbose=verbose, **kwargs)

    def read_dgn_shp_by_date(self, pcd_date=None, update=False, confirmation_required=True,
                             verbose=False, rm_dgn_shp_files=False, **kwargs):
        """
        Read DGN-converted shapefiles for a specific date of the point cloud data.

        :param pcd_date: Date of the point cloud data, e.g., ``'201910'``, ``'202004'``.
            When ``pcd_date=None`` (default), the function reads shapefiles for all available
            "ECM8_10_66_*_20200625" DGN data.
        :type pcd_date: str | None
        :param update: Whether to re-convert and read the original DGN data; defaults to ``False``.
        :type update: bool
        :param confirmation_required: Whether to ask for confirmation to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :param rm_dgn_shp_files: Whether to remove the converted shapefiles; defaults to ``False``.
        :type rm_dgn_shp_files: bool
        :param kwargs: [Optional] additional parameters for the function
            :func:`~src.utils.dgn.dgn2shp`.
        :return: Shapefile data of "ECM8_10_66_*_20200625".
        :rtype: dict

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> dgn_shp_files = pcd.read_dgn_shp_by_date()
            >>> type(dgn_shp_files)
            dict
            >>> list(dgn_shp_files.keys())
            ['201910', '202004']
            >>> dgn_shp_data_202004 = dgn_shp_files['202004']
            >>> type(dgn_shp_data_202004)
            dict
            >>> list(dgn_shp_data_202004.keys())
            ['Annotation', 'MultiPatch', 'Point', 'Polygon', 'Polyline']
            >>> dgn_shp_data_202004['Annotation'].shape
            (152628, 47)
            >>> dgn_shp_data_202004['Annotation'].head()
                   Entity  ...                                geometry
            0        Text  ...  POINT Z (340131.462 674089.321 33.355)
            1        Text  ...  POINT Z (340131.931 674089.495 33.353)
            2        Text  ...  POINT Z (340132.868 674089.844 33.350)
            3        Text  ...  POINT Z (340133.805 674090.192 33.345)
            4        Text  ...  POINT Z (340134.742 674090.542 33.341)
            [5 rows x 47 columns]
            >>> dgn_shp_data_202004['Polyline'].shape
            (310118, 26)
            >>> dgn_shp_data_202004['Polyline'].head()
                           Entity  ...                                           geometry
            0          LineString  ...  LINESTRING Z (340131.462 674089.321 33.345, 34...
            1          LineString  ...  LINESTRING Z (340131.931 674089.495 33.343, 34...
            2          LineString  ...  LINESTRING Z (340132.868 674089.844 33.340, 34...
            3          LineString  ...  LINESTRING Z (340133.805 674090.192 33.335, 34...
            4          LineString  ...  LINESTRING Z (340134.742 674090.542 33.331, 34...
            [5 rows x 26 columns]
        """

        if pcd_date:
            shp_file_paths = [
                [cd(self.dgn_dir_(pcd_date), self.dgn_shp_filename_(pcd_date, x))
                 for x in dgn_shapefiles()]]
        else:
            shp_file_paths = [
                [cd(self.dgn_dir_(d), self.dgn_shp_filename_(d, x)) for x in dgn_shapefiles()]
                for d in self.data_dates]

        dgn_shp_dat = []
        for shp_paths in shp_file_paths:

            comm_dir = os.path.commonpath(shp_paths)
            dat_date = os.path.basename(comm_dir)
            path_to_pickle = cd(comm_dir, self.dgn_filename_(dat_date).replace(".dgn", ".pkl"))

            if os.path.isfile(path_to_pickle) and not update:
                shp_dat = load_data(path_to_pickle)

            else:
                if not all(os.path.isfile(x) for x in shp_paths):
                    self.dgn2shp(
                        pcd_date=dat_date, confirmation_required=confirmation_required,
                        verbose=verbose, **kwargs)

                # parsed_shp = []
                # for f in shp_paths:
                #     if os.path.isfile(f):
                #         dat = read_shp_file(f, engine='geopandas')
                #         # columns_to_drop = [x for x in dat.columns if len(dat[x].unique()) == 1]
                #         columns_to_drop = [
                #             'Layer', 'LvlDesc', 'Class', 'CadModel', 'DocName', 'DocPath',
                #             'DocType', 'DocVer']
                #         dat.drop(labels=columns_to_drop, axis=1, inplace=True)
                #     else:
                #         dat = None
                #     parsed_shp.append(dat)
                #
                # shp_dat = dict(zip(dgn_shapefiles(), parsed_shp))

                shp_dat = read_dgn_shapefile(comm_dir, self.dgn_filename_(dat_date))

                if rm_dgn_shp_files:
                    for f in shp_paths:
                        for shp_f in glob.glob(os.path.splitext(f)[0] + ".*"):
                            os.remove(shp_f)

                save_data(shp_dat, path_to_file=path_to_pickle, verbose=verbose)

            dgn_shp_dat.append(shp_dat)

        if pcd_date and len(dgn_shp_dat) == 1:
            dgn_shp_data = {pcd_date: dgn_shp_dat[0]}
        else:  # len(dgn_shp_dat) > 1
            dgn_shp_data = dict(zip(self.data_dates, dgn_shp_dat))

        return dgn_shp_data

    def read_dgn_shp(self, update=False, verbose=False, **kwargs):
        """
        Read all available DGN-converted shapefiles from a local directory.

        :param update: Whether to re-convert and read the original DGN data; defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the function
            :func:`~src.utils.dgn.dgn2shp`.
        :return: Data of DGN-converted shapefiles or ``None`` if no files are found.
        :rtype: dict | None

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> dgn_shp = pcd.read_dgn_shp()
            >>> type(dgn_shp)
            dict
            >>> list(dgn_shp.keys())
            ['Annotation', 'MultiPatch', 'Polygon', 'Polyline', 'Point']
            >>> dgn_shp['Annotation'].shape
            (305249, 49)
            >>> dgn_shp['Annotation'].head()
                    Year  Month  ... TxtMemo                                geometry
            0       2019     10  ...       1  POINT Z (340131.735 674089.424 33.353)
            1       2019     10  ...       2  POINT Z (340132.204 674089.599 33.352)
            2       2019     10  ...       3  POINT Z (340133.141 674089.947 33.348)
            3       2019     10  ...       4  POINT Z (340134.078 674090.296 33.344)
            4       2019     10  ...       5  POINT Z (340135.015 674090.645 33.340)
            [5 rows x 49 columns]
            >>> dgn_shp['Polyline'].shape
            (615378, 28)
            >>> dgn_shp['Polyline'].head()
                    Year  Month  ... QrotZ                                           geometry
            0       2019     10  ...   0.0  LINESTRING Z (340131.735 674089.424 33.343, 34...
            1       2019     10  ...   0.0  LINESTRING Z (340132.204 674089.599 33.342, 34...
            2       2019     10  ...   0.0  LINESTRING Z (340133.141 674089.947 33.338, 34...
            3       2019     10  ...   0.0  LINESTRING Z (340134.078 674090.296 33.334, 34...
            4       2019     10  ...   0.0  LINESTRING Z (340135.015 674090.645 33.330, 34...
            [5 rows x 28 columns]
        """

        path_to_pickle = cd(self.dgn_dir_(""), "pcd_dgn_shp.pkl")

        if os.path.isfile(path_to_pickle) and not update:
            dgn_shp_data = load_data(path_to_pickle)

        else:
            # print("Reading DGN-shapefiles of point cloud data:") if verbose else ""

            try:
                dgn_shp_data_ = []

                for pcd_date in self.data_dates:
                    dgn_shp_dat = self.read_dgn_shp_by_date(
                        pcd_date=pcd_date, update=update, verbose=verbose, **kwargs)[pcd_date]

                    pcd_date_ = datetime.datetime.strptime(pcd_date, '%Y%m').date()

                    for k in dgn_shp_dat:
                        dgn_shp_dat[k].insert(0, 'Year', pcd_date_.year)
                        dgn_shp_dat[k].insert(1, 'Month', pcd_date_.month)

                    dgn_shp_data_.append(dgn_shp_dat)

                dgn_shp_data = {
                    k: pd.concat([d.get(k) for d in dgn_shp_data_], ignore_index=True)
                    for k in set().union(*dgn_shp_data_)}

                # if verbose:
                #     print("Done.")

                save_data(dgn_shp_data, path_to_file=path_to_pickle, verbose=verbose)

            except Exception as e:
                _print_failure_msg(e)
                dgn_shp_data = None

        return dgn_shp_data

    def import_dgn_shp(self, confirmation_required=True, verbose=True, rm_dgn_shp_files=False,
                       **kwargs):
        """
        Import shapefile data (converted from DGN data) into the project database.

        :param confirmation_required: Whether to ask for confirmation to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int
        :param rm_dgn_shp_files: Whether to remove the converted shapefiles after importing;
            defaults to ``False``.
        :type rm_dgn_shp_files: bool
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.import_data`_.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> pcd.import_dgn_shp(if_exists='replace')
            To import DGN-shapefiles into the schema "PCD"?
             [No]|Yes: yes
            Importing the data:
                Annotation ... Done.
                Polyline ... Done.

        .. figure:: ../_images/pcd_dgn_annotation_tbl.*
            :name: pcd_dgn_annotation_tbl
            :align: center
            :width: 100%

            Snapshot of the "PCD"."DGN_Shapefile_Annotation" table.

        .. figure:: ../_images/pcd_dgn_polyline_tbl.*
            :name: pcd_dgn_polyline_tbl
            :align: center
            :width: 100%

            Snapshot of the "PCD"."DGN_Shapefile_Polyline" table.
        """

        cfm_msg = f'To import DGN-shapefiles into the schema "{self.SCHEMA_NAME}"?\n'
        if confirmed(cfm_msg, confirmation_required=confirmation_required):

            dgn_shp = self.read_dgn_shp(rm_dgn_shp_files=rm_dgn_shp_files)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing the data: ")

            for k, shp_dat in dgn_shp.items():
                if not shp_dat.empty:
                    print("\t{}".format(k), end=" ... ") if verbose else ""

                try:
                    if not shp_dat.empty:
                        dat_ = shp_dat.copy()
                        dat_['geometry'] = dat_['geometry'].map(lambda x: x.wkt)

                        self.db_instance.import_data(
                            data=dat_, table_name=self.dgn_shp_table_name_(k),
                            schema_name=self.SCHEMA_NAME, method=self.db_instance.psql_insert_copy,
                            confirmation_required=False, **kwargs)

                        if verbose:
                            print("Done.")

                except Exception as e:
                    _print_failure_msg(e)

    @staticmethod
    def flatten_dgn_pcd(dgn_pl_dat):
        """
        Flatten the polyline geometry data of the DGN-converted shapefile.

        :param dgn_pl_dat: Polyline geometry data of the DGN-converted shapefile.
        :type dgn_pl_dat: pandas.DataFrame
        :return: Flattened point coordinates.
        :rtype: numpy.ndarray

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> from src.utils.general import TrackFixityDB
            >>> from pyhelpers.settings import np_preferences
            >>> project_db = TrackFixityDB()
            >>> pcd = PCD(db_instance=project_db)
            >>> tbl_name = pcd.dgn_shp_table_name_('Polyline')
            >>> query = f'SELECT * FROM "PCD"."{tbl_name}" WHERE "Year"=2020 AND "Month"=4 LIMIT 5'
            >>> example_dat = pcd.db_instance.read_sql_query(query)
            >>> example_dat.head()
               Year  Month  ... QrotZ                                           geometry
            0  2020      4  ...     0  LINESTRING Z (340183.475 674108.682 33.237, 34...
            1  2020      4  ...     0  LINESTRING Z (340184.412 674109.031 33.235, 34...
            2  2020      4  ...     0  LINESTRING Z (340185.349 674109.38 33.232, 340...
            3  2020      4  ...     0  LINESTRING Z (340233.146 674127.169 33.117, 34...
            4  2020      4  ...     0  LINESTRING Z (340234.083 674127.518 33.114, 34...
            [5 rows x 28 columns]
            >>> np_preferences()
            >>> pcd.flatten_dgn_pcd(example_dat)
            array([[340183.4750, 674108.6820, 33.2370],
                   [340183.4750, 674108.6820, 33.2550],
                   [340184.4120, 674109.0310, 33.2350],
                   [340184.4120, 674109.0310, 33.2530],
                   [340185.3490, 674109.3800, 33.2320],
                   [340185.3490, 674109.3800, 33.2500],
                   [340233.1460, 674127.1690, 33.1170],
                   [340233.1460, 674127.1690, 33.1350],
                   [340234.0830, 674127.5180, 33.1140],
                   [340234.0830, 674127.5180, 33.1320]])
        """

        flat_geom_coords_list = flatten_geometry(dgn_pl_dat.geometry, as_array=True)

        return flat_geom_coords_list

    def load_dgn_shp(self, pcd_date=None, layer_name=None):
        """
        Load data of the DGN-converted shapefile from the project database.

        :param pcd_date: Date of the DGN data; defaults to ``None``.
        :type pcd_date: str | int | None
        :param layer_name: Layer name of the DGN-converted shapefile; defaults to ``None``.
        :type layer_name: str | None
        :return: Raw data of the DGN-converted shapefile.
        :rtype: tuple

        .. note::

            The returned tuple contains two dictionaries:
              - ``'Annotation'``: The ``'Annotation'`` layer.
              - ``'Polyline'``: The ``'Complex Chain'`` and ``'LineString'`` entities.

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> dgn_pl, dgn_pl_pcd = pcd.load_dgn_shp()
            >>> type(dgn_pl)
            dict
            >>> list(dgn_pl.keys())
            ['Annotation', 'Polyline']
            >>> dgn_pl['Annotation'].shape
            (305249, 49)
            >>> dgn_pl['Polyline'].shape
            (615378, 28)
            >>> type(dgn_pl_pcd)
            dict
            >>> list(dgn_pl_pcd.keys())
            ['Annotation', 'Polyline']
            >>> dgn_pl_pcd['Annotation'].shape
            (305249, 3)
            >>> list(dgn_pl_pcd['Polyline'].keys())
            ['Complex Chain', 'LineString']
            >>> # 'Polyline' data of April 2020
            >>> dgn_pl, dgn_pl_pcd = pcd.load_dgn_shp(pcd_date='202004', layer_name='Polyline')
            >>> dgn_pl.shape
            (310118, 28)
            >>> type(dgn_pl_pcd)
            dict
            >>> list(dgn_pl_pcd.keys())
            ['Complex Chain', 'LineString']
            >>> dgn_pl_pcd['Complex Chain'].shape
            (1068396, 3)
            >>> dgn_pl_pcd['LineString'].shape
            (620208, 3)
            >>> # 'Annotation' data of April 2020
            >>> dgn_annot, dgn_annot_pcd = pcd.load_dgn_shp('202004', layer_name='Annotation')
            >>> dgn_annot.shape
            (152628, 49)
            >>> dgn_annot_pcd.shape
            (152628, 3)
        """

        valid_layer_names = ('Annotation', 'Polyline')

        if layer_name is None:
            layer_name_ = valid_layer_names
            dgn_lyr_list, dgn_pcd_list = [], []
            for lyr_name in layer_name_:
                dgn_lyr_, dgn_pcd_ = self.load_dgn_shp(pcd_date=pcd_date, layer_name=lyr_name)
                dgn_lyr_list.append(dgn_lyr_)
                dgn_pcd_list.append(dgn_pcd_)

            dgn_lyr = dict(zip(layer_name_, dgn_lyr_list))
            dgn_pcd = dict(zip(layer_name_, dgn_pcd_list))

        else:
            layer_name_ = find_similar_str(layer_name, valid_layer_names)
            dgn_lyr_tbl_name = self.dgn_shp_table_name_(layer_name_)

            sql_query = f'SELECT * FROM "PCD"."{dgn_lyr_tbl_name}"'

            if pcd_date is not None:
                year, month = data_date_to_year_month(data_date=pcd_date)
                sql_query += f' WHERE "Year"={year} AND "Month"={month}'

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            dgn_lyr = self.db_instance.read_sql_query(sql_query=sql_query)

            multi_val_cols = [
                x for x in dgn_lyr.columns
                if len(dgn_lyr[x].unique()) > 1 and x not in (
                    'Year', 'Month', 'Elevation', 'geometry')]
            dgn_lyr.sort_values(by=multi_val_cols, ascending=False, inplace=True, ignore_index=True)

            if layer_name_ == 'Annotation':
                dgn_pcd = self.flatten_dgn_pcd(dgn_lyr)

            elif layer_name_ == 'Polyline':
                entity_names = ['Complex Chain', 'LineString']

                # 'Complex Chain' and 1-metre 'LineString'
                dgn_pl_cc, dgn_pl_ls = [x for _, x in dgn_lyr.groupby(dgn_lyr.Entity)]
                pl_cc_pcd, pl_ls_pcd = map(self.flatten_dgn_pcd, (dgn_pl_cc, dgn_pl_ls))

                dgn_pcd = dict(zip(entity_names, [pl_cc_pcd, pl_ls_pcd]))

            else:
                dgn_pcd = None

        return dgn_lyr, dgn_pcd

    # == KRDZ data =================================================================================

    def read_krdz_metadata(self, update=False, verbose=False):
        """
        Read the metadata for the KRDZ data from a local directory.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Column names and descriptions of the KRDZ data.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> krdz_schema_info = pcd.read_krdz_metadata()
            >>> krdz_schema_info.shape
            (27, 2)
        """

        path_to_krdz_schema = cd(self.DATA_DIR, self.KRDZ_SCHEMA_FILENAME + ".txt")
        path_to_pickle = path_to_krdz_schema.replace(".txt", ".pkl")

        if os.path.isfile(path_to_pickle) and not update:
            krdz_schema = load_data(path_to_pickle)

        else:
            krdz_schema = pd.read_csv(path_to_krdz_schema, sep="\t")

            save_data(krdz_schema, path_to_pickle, verbose=verbose)

        return krdz_schema

    def import_krdz_metadata(self, update=False, confirmation_required=True, verbose=True,
                             **kwargs):
        """
        Import metadata for the KRDZ data into the project database.

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

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> pcd.import_krdz_metadata()
            To import KRDZ metadata into the table "PCD"."KRDZ_Metadata"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/pcd_krdz_metadata_tbl.*
            :name: pcd_krdz_metadata_tbl
            :align: center
            :width: 100%

            Snapshot of the "PCD"."KRDZ_Metadata" table.
        """

        tbl_name = f'"{self.SCHEMA_NAME}"."{self.KRDZ_META_TABLE_NAME}"'
        cfm_msg = f"To import KRDZ metadata into the table {tbl_name}?\n"

        if confirmed(cfm_msg, confirmation_required=confirmation_required):

            krdz_schema = self.read_krdz_metadata(update=update, verbose=verbose)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing the data", end=" ... ")

            try:
                self.db_instance.import_data(
                    data=krdz_schema, schema_name=self.KRDZ_SCHEMA_NAME,
                    table_name=self.KRDZ_META_TABLE_NAME, method=self.db_instance.psql_insert_copy,
                    confirmation_required=False, **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e)

    def read_krdz_by_date(self, pcd_date, update=False, verbose=False):
        """
        Read a KRDZ file for a given date from a local directory.

        :param pcd_date: Date of the KRDZ file in the format 'YYYYMMDD'.
        :type pcd_date: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data from the KRDZ file as a pandas DataFrame, or ``None`` if no data is found.
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> krdz_file_201910 = pcd.read_krdz_by_date(pcd_date='201910')
            >>> krdz_file_201910.shape
            (152621, 27)
            >>> krdz_file_202004 = pcd.read_krdz_by_date(pcd_date='202004')
            >>> krdz_file_202004.shape
            (152628, 27)
        """

        path_to_krdz = cd(self.krdz_dir_(pcd_date), self.krdz_filename_(pcd_date))
        path_to_pickle = path_to_krdz + ".pkl"

        if os.path.isfile(path_to_pickle) and not update:
            krdz_data = load_data(path_to_pickle)

        else:
            if os.path.isfile(path_to_krdz):
                krdz_file_desc = self.read_krdz_metadata()
                column_names = krdz_file_desc['Name'].to_list()

                krdz_data = pd.read_csv(
                    path_to_krdz, header=None, sep=r'\s+', names=column_names, low_memory=False)

                if pcd_date == '201910':
                    left_columns = [x for x in krdz_data.columns if 'Left' in x]
                    right_columns = [y for y in krdz_data.columns if 'Right' in y]

                    for left_col, right_col in zip(left_columns, right_columns):
                        assert all('RunningEdge' in col for col in (left_col, right_col)) or all(
                            'TopOfRail' in col for col in (left_col, right_col))

                        krdz_data[[left_col, right_col]] = krdz_data[[right_col, left_col]]

                save_data(krdz_data, path_to_pickle, verbose=verbose)

            else:
                krdz_data = None

        return krdz_data

    def read_krdz(self, update=False, verbose=False):
        """
        Read a KRDZ file from a local directory.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data from the KRDZ file as a pandas DataFrame, or ``None`` if no data is found.
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> krdz_file = pcd.read_krdz()
            >>> krdz_file.shape
            (305249, 29)
        """

        path_to_pickle = cd(self.krdz_dir_(""), "krdz.pkl")

        if os.path.isfile(path_to_pickle) and not update:
            krdz_data = load_data(path_to_pickle)

        else:
            print("Reading KRDZ files of point cloud data:") if verbose else ""

            try:
                krdz_data_ = []
                for pcd_date in self.data_dates:
                    krdz_dat = self.read_krdz_by_date(
                        pcd_date=pcd_date, update=update, verbose=verbose)

                    if krdz_dat is not None:
                        pcd_date_ = datetime.datetime.strptime(pcd_date, '%Y%m').date()

                        krdz_dat.insert(0, 'Year', pcd_date_.year)
                        krdz_dat.insert(1, 'Month', pcd_date_.month)

                        krdz_data_.append(krdz_dat)

                krdz_data = pd.concat(krdz_data_, ignore_index=True)

                if verbose:
                    print("Done.")

                save_data(krdz_data, path_to_pickle, verbose=verbose)

            except Exception as e:
                _print_failure_msg(e)
                krdz_data = None

        return krdz_data

    def import_krdz(self, update=False, confirmation_required=True, verbose=True, **kwargs):
        """
        Import KRDZ data into the project database.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param confirmation_required: Whether to ask for confirmation to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.import_data`.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> pcd.import_krdz(if_exists='replace')
            To import KRDZ data of ECM8 into the table "PCD"."KRDZ"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/pcd_krdz_tbl.*
            :name: pcd_krdz_tbl
            :align: center
            :width: 100%

            Snapshot of the "PCD"."KRDZ" table.
        """

        # if isinstance(self.elr, (list, tuple)):
        #     elr_ = f' ({", ".join(self.elr)})'
        if isinstance(self.elr, str):
            elr_ = f' ({self.elr})'
        else:
            elr_ = ''
        dat_name = f"KRDZ data{elr_}"
        tbl_name = f'"{self.SCHEMA_NAME}"."{self.KRDZ_TABLE_NAME}"'

        cfm_msg = f"To import {dat_name} into the table {tbl_name}?\n"
        if confirmed(cfm_msg, confirmation_required=confirmation_required):

            krdz_data = self.read_krdz(update=update, verbose=verbose)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing the data", end=" ... ")

            try:
                self.db_instance.import_data(
                    data=krdz_data, schema_name=self.KRDZ_SCHEMA_NAME,
                    table_name=self.KRDZ_TABLE_NAME, method=self.db_instance.psql_insert_copy,
                    confirmation_required=False, **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e)

    def load_krdz(self, pcd_dates=None, **kwargs):
        """
        Load KRDZ data (for a given data date or dates) from the project database.

        :param pcd_dates: Date(s) of the point cloud data; defaults to ``None``.
        :type pcd_dates: str | list | None
        :param kwargs: [Optional] additional parameters of the method
            `pyhelpers.dbms.PostgreSQL.read_sql_query`_.
        :return: KRDZ data for the given date or dates.
        :rtype: pandas.DataFrame

        .. _`pyhelpers.dbms.PostgreSQL.read_sql_query`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.read_sql_query.html

        **Examples**::

            >>> from src.preprocessor import PCD
            >>> pcd = PCD()
            >>> krdz_dat_201910 = pcd.load_krdz('201910')
            >>> krdz_dat_201910.shape
            (152621, 29)
            >>> krdz_dat = pcd.load_krdz()
            >>> krdz_dat.shape
            (305249, 29)
        """

        if self.db_instance is None:
            self.db_instance = TrackFixityDB()

        sql_query = f'SELECT * FROM "{self.KRDZ_SCHEMA_NAME}"."{self.KRDZ_TABLE_NAME}"'

        # Data dates
        sql_query = add_sql_query_date_condition(sql_query, data_date=pcd_dates)

        krdz_data = self.db_instance.read_sql_query(sql_query=sql_query, **kwargs)

        return krdz_data
