"""
This module processes the KRDZ data to enable calculation/measurement of the track movement.
"""

import datetime
import functools
import gc
import itertools
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.ops
import shapely.wkt
from pyhelpers.dirs import cd
from pyhelpers.ops import confirmed
from pyhelpers.store import load_data, save_data, save_figure
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon
from sklearn.cluster import DBSCAN

from src.preprocessor import Track
from src.shaft.pcd_handler import PCDHandler
from src.utils.general import (TrackFixityDB, add_sql_query_date_condition,
                               add_sql_query_xy_condition, cd_docs_source, data_date_to_year_month,
                               find_valid_names, get_tile_xy)
from src.utils.geometry import (calculate_slope, find_shortest_path, make_a_polyline, offset_ls,
                                point_projected_to_line)


class KRDZGear(PCDHandler):
    """
    Further process the KRDZ data (within the point cloud category).
    """

    #: Descriptive name of the class.
    NAME: str = "KRDZ data"
    #: Railway directions, including 'up' (being towards a major location) and 'down'.
    DIRECTIONS: list = ['Up', 'Down']
    #: Different parts of rail head, e.g. left/right top of rail or running edge.
    ELEMENTS: list = [
        'LeftTopOfRail',
        'LeftRunningEdge',
        'RightTopOfRail',
        'RightRunningEdge',
        'Centre',
    ]

    def __init__(self, elr='ECM8', db_instance=None):
        """
        :param elr: Engineer's Line Reference; defaults to ``'ECM8'``.
        :type elr: str
        :param db_instance: PostgreSQL database instance; defaults to ``None``..
        :type db_instance: TrackFixityDB | None

        :ivar str elr: Engineer's Line Reference.
        :ivar Track trk: Instance of the class :class:`~src.preprocessor.track.Track`.

        **Examples**::

            >>> from src.shaft import KRDZGear
            >>> krdzg = KRDZGear()
            >>> krdzg.NAME
            "KRDZ data"
        """

        super().__init__(db_instance=db_instance)

        self.elr = elr

        self.trk = Track(db_instance=self.db_instance)

    def get_krdz_in_pcd_tile(self, xyz, tile_xy, pcd_date=None, radius=0.0):
        """
        Find all points that are within a specified tile and data date
        from a given set of geographic coordinates (or point data).

        :param xyz: Geographic coordinates (or point data).
        :type xyz: numpy.ndarray | LineString
        :param tile_xy: X and Y coordinates in reference to a tile for the point cloud data.
        :type tile_xy: tuple | list | str
        :param pcd_date: Date of the point cloud data; defaults to ``None``.
        :type pcd_date: str | int | None
        :param radius: Radius; defaults to ``0.0``.
        :type radius: float
        :return: All data within the given ``tile_xy`` and data date ``pcd_date``.
        :rtype: numpy.ndarray

        **Examples**::

            >>> from src.shaft import KRDZGear
            >>> from pyhelpers.settings import np_preferences
            >>> np_preferences()
            >>> krdzg = KRDZGear()
            >>> # Get KRDZ rail head data
            >>> krdz_data = krdzg.load_krdz(pcd_dates='201910')
            >>> krdz_data
                    Year  Month  ...  zRightRunningEdge  zRightTopOfRail
            0       2019     10  ...             33.341           33.356
            1       2019     10  ...             33.335           33.351
            2       2019     10  ...             33.331           33.346
            3       2019     10  ...             33.327           33.343
            4       2019     10  ...             33.324           33.340
            ...      ...    ...  ...                ...              ...
            152616  2019     10  ...             33.316           33.332
            152617  2019     10  ...             33.319           33.335
            152618  2019     10  ...             33.322           33.337
            152619  2019     10  ...             33.324           33.339
            152620  2019     10  ...             33.327           33.342
            [152621 rows x 29 columns]
            >>> left_top_cols = ['xLeftTopOfRail', 'yLeftTopOfRail', 'zLeftTopOfRail']
            >>> left_top_201910 = krdz_data[left_top_cols].to_numpy()
            >>> left_top_201910.shape
            (152621, 3)
            >>> lt_201910 = krdzg.get_krdz_in_pcd_tile(left_top_201910, tile_xy=(340500, 674200))
            >>> lt_201910.shape
            (214, 3)
        """

        if tile_xy is not None:
            # Make a matplotlib.path.Path object
            tile_polyline_path = self.get_pcd_tile_mpl_path(tile_xy=tile_xy, pcd_date=pcd_date)

            # Whether the area enclosed by the path contains the given points
            if isinstance(xyz, LineString):
                xyz_dat = np.array(xyz.coords)[:, [0, 1]]
            else:
                xyz_dat = xyz[:, [0, 1]]
            mask = tile_polyline_path.contains_points(xyz_dat, radius=radius)

            xyz_ = xyz[mask, :]

        else:
            xyz_ = xyz

        return xyz_

    def load_pcd_krdz(self, tile_xy=None, pcd_date=None):
        """
        Get (X, Y, Z) coordinates of the left and right tops, as well as the running edges,
        of the rail heads from the KRDZ data.

        :param tile_xy: X and Y coordinates in reference to a tile for the point cloud data;
            defaults to ``None``.
        :type tile_xy: tuple | list | str | None
        :param pcd_date: Date of the point cloud data; defaults to ``None``.
        :type pcd_date: str | int | None
        :return: (X, Y, Z) coordinates of the left top, left running edge, right top,
            right running edge and center line.
        :rtype: dict

        **Examples**::

            >>> from src.shaft import KRDZGear
            >>> krdzg = KRDZGear()
            >>> krdz_xyz_dat = krdzg.load_pcd_krdz(pcd_date='202004')
            >>> type(krdz_xyz_dat)
            dict
            >>> list(krdz_xyz_dat.keys())
            ['LeftTopOfRail',
             'LeftRunningEdge',
             'RightTopOfRail',
             'RightRunningEdge',
             'Centre']
            >>> krdz_xyz_dat['LeftTopOfRail']
            array([[340131.7340, 674088.6170, 33.3540],
                   [340132.2030, 674088.7910, 33.3530],
                   [340133.1410, 674089.1400, 33.3490],
                   [340134.0780, 674089.4890, 33.3440],
                   [340135.0150, 674089.8380, 33.3390],
                   ...,
                   [340136.4910, 674086.7520, 33.3330],
                   [340135.5540, 674086.4030, 33.3350],
                   [340134.6170, 674086.0530, 33.3370],
                   [340133.6800, 674085.7040, 33.3400],
                   [340132.7440, 674085.3540, 33.3440]])
            >>> krdz_xyz_dat['LeftTopOfRail'].shape
            (152628, 3)
            >>> krdz_xyz_dat['LeftRunningEdge']
            array([[340131.7220, 674088.6510, 33.3390],
                   [340132.1910, 674088.8260, 33.3380],
                   [340133.1280, 674089.1740, 33.3340],
                   [340134.0650, 674089.5230, 33.3290],
                   [340135.0020, 674089.8730, 33.3240],
                   ...,
                   [340136.4790, 674086.7860, 33.3180],
                   [340135.5420, 674086.4370, 33.3200],
                   [340134.6050, 674086.0870, 33.3220],
                   [340133.6680, 674085.7380, 33.3250],
                   [340132.7310, 674085.3890, 33.3290]])
            >>> krdz_xyz_dat['LeftRunningEdge'].shape
            (152628, 3)
            >>> krdz_xyz_dat['RightTopOfRail']
            array([[340131.1890, 674090.0240, 33.3550],
                   [340131.6570, 674090.1990, 33.3540],
                   [340132.5940, 674090.5470, 33.3500],
                   [340133.5320, 674090.8950, 33.3460],
                   [340134.4680, 674091.2440, 33.3420],
                   ...,
                   [340135.9440, 674088.1580, 33.3360],
                   [340135.0070, 674087.8080, 33.3380],
                   [340134.0700, 674087.4590, 33.3410],
                   [340133.1330, 674087.1090, 33.3430],
                   [340132.1960, 674086.7610, 33.3470]])
            >>> krdz_xyz_dat['RightTopOfRail'].shape
            (152628, 3)
            >>> krdz_xyz_dat['RightRunningEdge']
            array([[340131.2030, 674089.9900, 33.3390],
                   [340131.6710, 674090.1650, 33.3380],
                   [340132.6080, 674090.5130, 33.3350],
                   [340133.5450, 674090.8620, 33.3300],
                   [340134.4820, 674091.2110, 33.3270],
                   ...,
                   [340135.9580, 674088.1250, 33.3210],
                   [340135.0210, 674087.7750, 33.3230],
                   [340134.0840, 674087.4260, 33.3250],
                   [340133.1470, 674087.0760, 33.3280],
                   [340132.2100, 674086.7270, 33.3320]])
            >>> krdz_xyz_dat['RightRunningEdge'].shape
            (152628, 3)
            >>> krdz_xyz_dat['Centre']
            array([[340131.4620, 674089.3210, 33.3550],
                   [340131.9310, 674089.4950, 33.3530],
                   [340132.8680, 674089.8440, 33.3500],
                   [340133.8050, 674090.1920, 33.3450],
                   [340134.7420, 674090.5420, 33.3410],
                   ...,
                   [340136.2180, 674087.4550, 33.3350],
                   [340135.2810, 674087.1060, 33.3370],
                   [340134.3440, 674086.7570, 33.3390],
                   [340133.4070, 674086.4070, 33.3420],
                   [340132.4700, 674086.0580, 33.3450]])
            >>> krdz_xyz_dat['Centre'].shape
            (152628, 3)
        """

        krdz_data = self.load_krdz(pcd_dates=pcd_date)

        labels = [
            'LeftTopOfRail',
            'LeftRunningEdge',
            'RightTopOfRail',
            'RightRunningEdge',
            'Centre',
        ]

        krdz_xyz_ = []

        for label in labels:
            col_names = [col for col in krdz_data.columns if label in col]
            xyz = krdz_data[col_names].to_numpy()
            krdz_xyz_.append(self.get_krdz_in_pcd_tile(xyz=xyz, tile_xy=tile_xy, pcd_date=pcd_date))

        krdz_xyz = dict(zip(labels, krdz_xyz_))

        return krdz_xyz

    def view_pcd_krdz(self, tile_xy, pcd_date, projection='3d', cmap_name='tab10', add_title=False,
                      save_as=None, dpi=600, verbose=False, **kwargs):
        # noinspection PyShadowingNames
        """
        Visualise original KRDZ data of the rail heads of point cloud data.

        :param tile_xy: X and Y coordinates in reference to a tile for the point cloud data.
        :type tile_xy: tuple | list | str
        :param pcd_date: Date of the point cloud data.
        :type pcd_date: str | int
        :param projection: Projection type of the subplot; defaults to ``'3d'``.
        :type projection: str | None
        :param cmap_name: Name of a matplotlib color map; defaults to ``'tab10'``.
        :type cmap_name: str | None
        :param add_title: Whether to add a title to the plot; defaults to ``False``.
        :type add_title: bool
        :param save_as: File format that the view is saved as; defaults to ``None``.
        :type save_as: str | list | None
        :param dpi: DPI for saving image; defaults to ``600``.
        :type dpi: int | None
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the function
            `matplotlib.pyplot.scatter`_.

        .. _`matplotlib.pyplot.scatter`:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

        **Examples**::

            >>> from src.shaft import KRDZGear
            >>> from pyhelpers.settings import mpl_preferences
            >>> mpl_preferences(backend='TkAgg')
            >>> krdzg = KRDZGear()
            >>> tile_xy = (340500, 674200)

        *October 2019*::

            >>> # 3D view
            >>> # krdzg.view_pcd_krdz(tile_xy, '201910', s=5, save_as=".svg", verbose=True)
            >>> krdzg.view_pcd_krdz(tile_xy, '201910', projection='3d', s=5)

        .. figure:: ../_images/krdzg_view_pcd_krdz_x340500y674200_201910_original_3d.*
            :name: krdzg_view_pcd_krdz_x340500y674200_201910_original_3d
            :align: center
            :width: 100%

            Original KRDZ data (3D) of Tile (340500, 674200) in October 2019.

        .. code-block:: python

            >>> # 2D plot - Vertical view
            >>> # krdzg.view_pcd_krdz(tile_xy, '201910', None, s=5, save_as=".svg", verbose=True)
            >>> krdzg.view_pcd_krdz(tile_xy, pcd_date='201910', projection=None, s=5)

        .. figure:: ../_images/krdzg_view_pcd_krdz_x340500y674200_201910_original.*
            :name: krdzg_view_pcd_krdz_x340500y674200_201910_original
            :align: center
            :width: 100%

            Original KRDZ data of Tile (340500, 674200) in October 2019.

        *April 2020*::

            >>> # 3D view
            >>> # krdzg.view_pcd_krdz(tile_xy, '202004', s=5, save_as=".svg", verbose=True)
            >>> krdzg.view_pcd_krdz(tile_xy, pcd_date='202004', projection='3d', s=5)

        .. figure:: ../_images/krdzg_view_pcd_krdz_x340500y674200_202004_original_3d.*
            :name: krdzg_view_pcd_krdz_x340500y674200_202004_original_3d
            :align: center
            :width: 100%

            Original KRDZ data (3D) of Tile (340500, 674200) in April 2020.

        .. code-block:: python

            >>> # 2D plot - Vertical view
            >>> # krdzg.view_pcd_krdz(tile_xy, '202004', None, s=5, save_as=".svg", verbose=True)
            >>> krdzg.view_pcd_krdz(tile_xy, pcd_date='202004', projection=None, s=5)

        .. figure:: ../_images/krdzg_view_pcd_krdz_x340500y674200_202004_original.*
            :name: krdzg_view_pcd_krdz_x340500y674200_202004_original
            :align: center
            :width: 100%

            Original KRDZ data of Tile (340500, 674200) in April 2020.
        """

        krdz_rail_head = self.load_pcd_krdz(tile_xy=tile_xy, pcd_date=pcd_date)

        colours = plt.colormaps.get_cmap(cmap_name).colors

        markers = [
            'x' if 'Edge' in label else ('o' if 'Top' in label else '^')
            for label in krdz_rail_head.keys()]

        zip_xyz_data = zip(krdz_rail_head.items(), colours[:5], markers)

        # Make a plot of the sample KRDZ data
        if projection:
            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(projection=projection)
            for (label, xyz), colour, marker in zip_xyz_data:
                kwargs.update({'color': colour, 'label': label, 'marker': marker})
                ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], **kwargs)
            ax.tick_params(axis='z', which='major', pad=10)  # , labelsize=18
            ax.set_xlabel('Easting', fontsize=18, labelpad=12)
            ax.set_ylabel('Northing', fontsize=18, labelpad=14)
            ax.set_zlabel('Elevation', fontsize=18, labelpad=18)

        else:
            fig = plt.figure(figsize=(11, 5))
            ax = fig.add_subplot(aspect='equal', adjustable='box')
            for (label, xyz), colour, marker in zip_xyz_data:
                kwargs.update({'color': colour, 'label': label, 'marker': marker})
                ax.scatter(xyz[:, 0], xyz[:, 1], **kwargs)
            ax.grid()
            ax.set_xlabel('Easting', fontsize=18, labelpad=5)
            ax.set_ylabel('Northing', fontsize=18, labelpad=5)

        ax.tick_params(axis='both', which='major')  # , labelsize=18
        ax.ticklabel_format(useOffset=False)

        tile_x, tile_y = get_tile_xy(tile_xy=tile_xy)

        if add_title:
            title_date = datetime.datetime.strptime(str(pcd_date), '%Y%m').strftime('%B %Y')
            ax.set_title(f"KRDZ data within Tile{tile_x, tile_y} in {title_date}.")

        ax.legend(loc='best', numpoints=1, ncol=1, fontsize=15)

        fig.tight_layout()
        if projection:
            fig.subplots_adjust(left=0, bottom=0, right=0.92, top=1)

        if save_as:
            suffix = f"_{projection}" if projection else ""
            fig_filename = f"krdzg_view_pcd_krdz_x{tile_x}y{tile_y}_{pcd_date}_original{suffix}"
            for save_as_ in {save_as, ".svg", ".pdf"}:
                path_to_fig = cd_docs_source("_images", fig_filename + save_as_)
                save_figure(fig, path_to_fig, dpi=dpi, transparent=True, verbose=verbose)

    @staticmethod
    def get_tiles_convex_hull(pcd_tiles, as_array=False):
        """
        Get a representation of the smallest convex polygon containing all the tiles
        for the point cloud data.

        :param pcd_tiles: The tiles for the point cloud data.
        :type pcd_tiles: numpy.ndarray | pandas.Series | pandas.DataFrame
        :param as_array: Whether to return the polygon as an array of vertices; 
            defaults to ``False``.
        :type as_array: bool
        :return: The smallest convex polygon containing all the tiles for the point cloud data.
        :rtype: Polygon | numpy.ndarray

        **Examples**::

            >>> from src.shaft import KRDZGear
            >>> krdzg = KRDZGear()
            >>> pcd_tiles_metadata_201910 = krdzg.load_tiles(pcd_date='201910')
            >>> pcd_tiles_metadata_201910.head()
               Year  ...                                        Tile_LonLat
            0  2019  ...  POLYGON ((-2.960910975517333 55.95617100494877...
            1  2019  ...  POLYGON ((-2.960888715229412 55.95527265723062...
            2  2019  ...  POLYGON ((-2.012726922542862 55.77482976892926...
            3  2019  ...  POLYGON ((-2.009538521884446 55.77393156353337...
            4  2019  ...  POLYGON ((-2.01113257284158 55.77393142614289,...
            [5 rows x 6 columns]
            >>> pcd_tiles_201910 = pcd_tiles_metadata_201910['Tile_XY']
            >>> pcd_tiles_201910.head()
            0    POLYGON ((340100 674100, 340100 674200, 340200...
            1    POLYGON ((340100 674000, 340100 674100, 340200...
            2    POLYGON ((399300 653500, 399300 653600, 399400...
            3    POLYGON ((399500 653400, 399500 653500, 399600...
            4    POLYGON ((399400 653400, 399400 653500, 399500...
            Name: Tile_XY, dtype: object
            >>> convex_hull = krdzg.get_tiles_convex_hull(pcd_tiles=pcd_tiles_201910)
            >>> type(convex_hull)
            shapely.geometry.polygon.Polygon
            >>> print(convex_hull.wkt)
            POLYGON ((399300 653400, 340100 674000, 340100 674300, 348300 678500, 349300 679000, ...
            >>> convex_hull_arr = krdzg.get_tiles_convex_hull(pcd_tiles_201910, as_array=True)
            >>> type(convex_hull_arr)
            numpy.ndarray
            >>> convex_hull_arr.shape
            (20, 2)

        **Illustration**::

            import matplotlib.pyplot as plt
            import numpy as np
            import shapely.wkt
            from pyhelpers.settings import mpl_preferences

            mpl_preferences(backend='TkAgg')

            fig = plt.figure(constrained_layout=True, figsize=(11, 5))
            ax = fig.add_subplot()
            ax.set_aspect(aspect='equal', adjustable='box')

            colours = plt.get_cmap('tab10').colors

            for tile in pcd_tiles_201910.map(shapely.wkt.loads):
                xs, ys = zip(*tile.exterior.coords)
                ax.plot(xs, ys, color=colours[0])
            ax.scatter([], [], marker='s', facecolors='none', edgecolors=colours[0], label='Tile')

            ch_xs, ch_ys = convex_hull_arr[:, 0], convex_hull_arr[:, 1]
            ax.plot(ch_xs, ch_ys, color=colours[1], label='Convex hull for the tiles')

            ax.legend()

            xmi, xma, ymi, yma = map(lambda x: int(x//10000*10000), ax.get_xlim() + ax.get_ylim())
            ax.xaxis.set_ticks(range(xmi + 10000, xma + 10000, 10000))
            ax.yaxis.set_ticks(range(ymi + 5000, yma + 5000, 5000))

            ax.set_xlabel('Easting', fontsize=13, labelpad=5)
            ax.set_ylabel('Northing', fontsize=13, labelpad=5)

            # from pyhelpers.store import save_figure
            #
            # fig_pathname = "docs/source/_images/krdzg_get_tiles_convex_hull_demo"
            # save_figure(fig, f"{fig_pathname}.svg", verbose=True)
            # save_figure(fig, f"{fig_pathname}.pdf", verbose=True)
            #
            # fig_pathname = "docs/source/_images/krdzg_get_tiles_convex_hull_demo_zoomed_in"
            # save_figure(fig, f"{fig_pathname}.svg", verbose=True)
            # save_figure(fig, f"{fig_pathname}.pdf", verbose=True)

        .. figure:: ../_images/krdzg_get_tiles_convex_hull_demo.*
            :name: krdzg_get_tiles_convex_hull_demo
            :align: center
            :width: 100%

            Convex hull for all tiles for the point cloud data.

        .. figure:: ../_images/krdzg_get_tiles_convex_hull_demo_zoomed_in.*
            :name: krdzg_get_tiles_convex_hull_demo_zoomed_in
            :align: center
            :width: 100%

            (Zoomed-in) Convex hull for all tiles for the point cloud data.
        """

        # e.g. pcd_tiles = tm.PCD.load_tiles(pcd_date='201910').Tile_XY
        if isinstance(pcd_tiles, pd.Series):
            try:
                pcd_tiles_ = pcd_tiles.map(shapely.wkt.loads).tolist()
            except (AssertionError, ValueError):
                pcd_tiles_ = pcd_tiles.tolist()
        elif isinstance(pcd_tiles, pd.DataFrame):
            assert pcd_tiles.shape[1] <= 2
            pcd_tiles_ = pcd_tiles.to_numpy()
        else:
            pcd_tiles_ = pcd_tiles

        try:
            tiles_polygon = MultiPolygon(pcd_tiles_)
        except ValueError:
            tiles_polygon = Polygon(pcd_tiles_)

        tiles_convex_hull = tiles_polygon.convex_hull

        if as_array:
            tiles_convex_hull = np.array(tiles_convex_hull.exterior.coords).astype(int)

        return tiles_convex_hull

    @staticmethod
    def get_key_reference(tiles_convex_hull, obj_attr=None, as_array=False):
        """
        Get a main reference object for classifying the KRDZ data between up and down directions.

        :param tiles_convex_hull: Convex hull of the tiles for the point cloud data.
        :type tiles_convex_hull: Polygon
        :param obj_attr: Attribute of the reference object;
            options include ``'centroid'`` (for the centroid of the object) and
            ``'rep'`` (for the representative point of the object);
            when ``obj_attr=None`` (default), the object remains its default type as a linestring.
        :type obj_attr: str | None
        :param as_array: Whether to convert the geometry object to array type;
            defaults to ``False``.
        :type as_array: bool
        :return: A reference point for identifying up and down directions.
        :rtype: LineString | Point | numpy.ndarray

        **Examples**::

            >>> from src.shaft import KRDZGear
            >>> krdzg = KRDZGear()
            >>> pcd_tiles_metadata_201910 = krdzg.load_tiles(pcd_date='201910')
            >>> pcd_tiles_metadata_201910.head()
               Year  ...                                        Tile_LonLat
            0  2019  ...  POLYGON ((-2.960910975517333 55.95617100494877...
            1  2019  ...  POLYGON ((-2.960888715229412 55.95527265723062...
            2  2019  ...  POLYGON ((-2.012726922542862 55.77482976892926...
            3  2019  ...  POLYGON ((-2.009538521884446 55.77393156353337...
            4  2019  ...  POLYGON ((-2.01113257284158 55.77393142614289,...
            [5 rows x 6 columns]
            >>> pcd_tiles_201910 = pcd_tiles_metadata_201910['Tile_XY']
            >>> pcd_tiles_201910.head()
            0    POLYGON ((340100 674100, 340100 674200, 340200...
            1    POLYGON ((340100 674000, 340100 674100, 340200...
            2    POLYGON ((399300 653500, 399300 653600, 399400...
            3    POLYGON ((399500 653400, 399500 653500, 399600...
            4    POLYGON ((399400 653400, 399400 653500, 399500...
            Name: Tile_XY, dtype: object
            >>> pcd_tiles_convex_hull = krdzg.get_tiles_convex_hull(pcd_tiles_201910)
            >>> print(pcd_tiles_convex_hull.wkt)
            POLYGON ((399300 653400, 340100 674000, 340100 674300, 348300 678500, 349300 679000, ...
            >>> ref_line = krdzg.get_key_reference(pcd_tiles_convex_hull)
            >>> type(ref_line)
            shapely.geometry.linestring.LineString
            >>> print(ref_line.wkt)
            LINESTRING (399300 653400, 340100 674000)
            >>> ref_pt = krdzg.get_key_reference(pcd_tiles_convex_hull, obj_attr='centroid')
            >>> type(ref_pt)
            shapely.geometry.point.Point
            >>> print(ref_pt.wkt)
            POINT (370139.9108845997 668965.4040025512)
            >>> ref_rep = krdzg.get_key_reference(pcd_tiles_convex_hull, obj_attr='rep')
            >>> type(ref_rep)
            shapely.geometry.point.Point
            >>> print(ref_rep.wkt)
            POINT (372007.0360450842 667600)

        **Illustration**::

            import matplotlib.pyplot as plt
            import numpy as np
            from pyhelpers.settings import mpl_preferences

            mpl_preferences(backend='TkAgg')

            fig = plt.figure(constrained_layout=True, figsize=(11, 5))
            ax = fig.add_subplot()
            ax.set_aspect(aspect='equal', adjustable='box')

            colours = plt.get_cmap('tab10').colors

            convex_hull_arr = np.array(pcd_tiles_convex_hull.exterior.coords)
            ax.plot(
                convex_hull_arr[:, 0], convex_hull_arr[:, 1], color=colours[0], zorder=2,
                label='Convex hull for all tiles\\n' \\
                      '(The area based on which the key reference objects are determined).')

            ref_line_coords = np.array(ref_line.coords)
            ax.plot(
                ref_line_coords[:, 0], ref_line_coords[:, 1], color=colours[1], linewidth=5,
                label='Ref LineString', zorder=1)
            ax.scatter(ref_pt.x, ref_pt.y, color=colours[2], label='Ref Centroid')
            ax.scatter(ref_rep.x, ref_rep.y, color=colours[3], label='Ref Rep')

            ax.legend()

            ax.set_xlabel('Easting', fontsize=13, labelpad=5)
            ax.set_ylabel('Northing', fontsize=13, labelpad=5)

            # from pyhelpers.store import save_figure
            #
            # fig_pathname = "docs/source/_images/krdzg_get_key_reference_demo"
            # save_figure(fig, f"{fig_pathname}.svg", verbose=True)
            # save_figure(fig, f"{fig_pathname}.pdf", verbose=True)

        .. figure:: ../_images/krdzg_get_key_reference_demo.*
            :name: krdzg_get_key_reference_demo
            :align: center
            :width: 100%

            Reference objects for classifying KRDZ data.
        """

        if obj_attr == 'centroid':
            key_ref_object = tiles_convex_hull.centroid

        elif obj_attr == 'rep':
            key_ref_object = tiles_convex_hull.representative_point()

        else:
            tiles_convex_hull_arr = np.array(tiles_convex_hull.exterior.coords).astype(int)

            if np.array_equal(tiles_convex_hull_arr[0], tiles_convex_hull_arr[-1]):
                start_point = tiles_convex_hull_arr[0]
                tiles_convex_hull_ = np.delete(tiles_convex_hull_arr, -1, axis=0)
            else:
                tiles_convex_hull_, counts = np.unique(
                    tiles_convex_hull_arr, axis=0, return_counts=True)
                start_point = tiles_convex_hull_[counts == 2]

            bounds_poly = make_a_polyline(
                points_sequence=tiles_convex_hull_, start_point=start_point)

            key_ref_object = LineString([np.array(bounds_poly.coords)[i] for i in [0, -1]])

        if as_array:
            key_ref_object = np.array(key_ref_object.coords)

        return key_ref_object

    @staticmethod
    def get_adjusted_ref_line_par(ref_line_par, tile_poly, aux_ref=None, centroid=False):
        """
        Adjust the position of a given reference line for clustering the original KRDZ data.

        :param ref_line_par: A reference line that is parallel to a pre-specified one.
        :type ref_line_par: LineString
        :param tile_poly: A tile.
        :type tile_poly: Polygon
        :param aux_ref: An auxiliary reference line; defaults to ``None``.
        :type aux_ref: LineString | None
        :param centroid: Whether to use the centroid of the adjusted reference line;
            defaults to ``False``.
        :type centroid: bool
        :return: Adjusted reference location based on the given reference line.
        :rtype: LineString | Point

        **Examples**::

            >>> from src.shaft import KRDZGear
            >>> krdzg = KRDZGear()
            >>> ref_objects = krdzg.get_reference_objects_for_krdz_clf('202004', par_dist=5000)
            >>> tiles_convex_hull, par_ref_ls, _ = ref_objects
            >>> type(par_ref_ls)
            shapely.geometry.linestring.LineString
            >>> print(par_ref_ls.wkt)
            LINESTRING (397656.7781152923 648677.7312827818, 338456.7781152923 669277.7312827818)
            >>> tile_x_y = (340100, 674000)
            >>> tile_polygon = krdzg.get_pcd_tile_polygon(tile_xy=tile_x_y)
            >>> type(tile_polygon)
            shapely.geometry.polygon.Polygon
            >>> print(tile_polygon.wkt)
            POLYGON ((340100 674000, 340100 674100, 340200 674100, 340200 674000, 340100 674000))
            >>> par_ref_ls_adj = krdzg.get_adjusted_ref_line_par(par_ref_ls, tile_poly=tile_polygon)

        **Illustration**::

            import matplotlib.pyplot as plt
            import numpy as np
            from pyhelpers.settings import mpl_preferences

            mpl_preferences(backend='TkAgg')

            fig = plt.figure(figsize=(11, 5), constrained_layout=True)
            ax = fig.add_subplot()
            ax.set_aspect('equal', adjustable='box')

            tch_arr = np.array(tiles_convex_hull.exterior.coords)
            ax.plot(tch_arr[:, 0], tch_arr[:, 1], zorder=3, label='Convex hull for all tiles')

            ref_line = krdzg.get_key_reference(tiles_convex_hull)
            rl_arr = np.array(ref_line.coords)
            ax.plot(rl_arr[:, 0], rl_arr[:, 1], lw=5, label='Ref line')

            tp_arr = np.array(tile_polygon.exterior.coords)
            ax.plot(tp_arr[:, 0], tp_arr[:, 1], label='Tile of (340100, 674000)')

            prl_arr = np.array(par_ref_ls.coords)
            ax.plot(prl_arr[:, 0], prl_arr[:, 1], lw=5, zorder=2, label='Ref line (Paralleled)')

            aprl_arr = np.array(par_ref_ls_adj.coords)
            ax.plot(aprl_arr[:, 0], aprl_arr[:, 1], lw=3, zorder=3, label='Ref line (Adjusted)')

            ax.legend()

            ax.set_xlabel('Easting', fontsize=13, labelpad=5)
            ax.set_ylabel('Northing', fontsize=13, labelpad=5)

            # from pyhelpers.store import save_figure
            #
            # fig_pathname = "docs/source/_images/krdzg_get_adj_ref_ls_par_demo"
            # save_figure(fig, f"{fig_pathname}.svg", verbose=True)
            # save_figure(fig, f"{fig_pathname}.pdf", verbose=True)
            #
            # fig_pathname = "docs/source/_images/krdzg_get_adj_ref_ls_par_demo_zoomed_in"
            # save_figure(fig, f"{fig_pathname}.svg", verbose=True)
            # save_figure(fig, f"{fig_pathname}.pdf", verbose=True)

        .. figure:: ../_images/krdzg_get_adj_ref_ls_par_demo.*
            :name: krdzg_get_adj_ref_ls_par_demo
            :align: center
            :width: 100%

            Adjusted reference line for classifying KRDZ data in Tile (340100, 674000).

        .. figure:: ../_images/krdzg_get_adj_ref_ls_par_demo_zoomed_in.*
            :name: krdzg_get_adj_ref_ls_par_demo_zoomed_in
            :align: center
            :width: 100%

            (Zoomed-in) Adjusted reference line for classifying KRDZ data in Tile (340100, 674000).
        """

        if isinstance(ref_line_par, np.ndarray):
            ref_ls_par = LineString(ref_line_par)
        else:
            ref_ls_par = ref_line_par

        ref_ls_par_ = point_projected_to_line(point=tile_poly.centroid, line=ref_ls_par)[1]

        if aux_ref is None:
            c = min(ref_ls_par.boundary.geoms, key=lambda x: x.distance(ref_ls_par_))
            ref_loc_adjusted = LineString([c, ref_ls_par_])
        else:
            a, b = sorted(ref_ls_par.boundary.geoms, key=lambda x: x.x)
            if calculate_slope(np.array(aux_ref.coords)) < 0:
                ref_line_par_adj_ = [a, ref_ls_par_]
            else:
                ref_line_par_adj_ = [ref_ls_par_, b]
            ref_loc_adjusted = LineString(ref_line_par_adj_)

        if centroid:
            ref_loc_adjusted = ref_loc_adjusted.centroid

        return ref_loc_adjusted

    @staticmethod
    def get_pcd_tile_tracks_shp(trk_shp, tile_poly):
        """
        Get track shapefiles with respect to a given tile (for point cloud data).

        :param trk_shp: A number of geometry objects of track lines.
        :type trk_shp: MultiLineString
        :param tile_poly: A tile.
        :type tile_poly: Polygon
        :return: Track lines that intersect or are near the given tile.
        :rtype: MultiLineString

        **Examples**::

            >>> from src.shaft import KRDZGear
            >>> from shapely.geometry import MultiLineString
            >>> krdzg = KRDZGear()
            >>> trk_shp_data = krdzg.trk.load_tracks_shp(elr=['ECM7', 'ECM8'])
            >>> track_shp = MultiLineString(trk_shp_data['geometry'].to_list())
            >>> type(track_shp)
            shapely.geometry.multilinestring.MultiLineString
            >>> len(track_shp.geoms)
            440
            >>> tile_xy = (380600, 665400)
            >>> tile_polygon = krdzg.get_pcd_tile_polygon(tile_xy=tile_xy)
            >>> type(tile_polygon)
            shapely.geometry.polygon.Polygon
            >>> print(tile_polygon.bounds)
            (380600.0, 665400.0, 380700.0, 665500.0)
            >>> track_lines = krdzg.get_pcd_tile_tracks_shp(track_shp, tile_poly=tile_polygon)
            >>> type(track_lines)
            shapely.geometry.multilinestring.MultiLineString
            >>> len(track_lines.geoms)
            2

        **Illustration**::

            import matplotlib.pyplot as plt
            import numpy as np
            from pyhelpers.settings import mpl_preferences

            mpl_preferences(backend='TkAgg')

            colours = plt.get_cmap('tab10').colors

            fig = plt.figure(figsize=(6, 6), constrained_layout=True)
            ax = fig.add_subplot(aspect='equal', adjustable='box')

            tile_poly_coords = np.array(tile_polygon.exterior.coords)
            ax.plot(
                tile_poly_coords[:, 0], tile_poly_coords[:, 1], color=colours[0],
                label='Tile (380600, 665400)')

            for ls in track_shp.geoms:
                ls_coords = np.array(ls.coords)
                ax.plot(ls_coords[:, 0], ls_coords[:, 1], color=colours[1])
            ax.plot([], [], color=colours[1], label='Track shapefiles')

            for ls in track_lines.geoms:
                ls_coords = np.array(ls.coords)
                ax.plot(ls_coords[:, 0], ls_coords[:, 1], color=colours[2], linewidth=3)
            ax.plot(
                [], [], color=colours[2], linewidth=3,
                label='Track shapefiles w.r.t. the Tile (380600, 665400)')

            xmi, xma, ymi, yma = map(lambda x: int(x//10000*10000), ax.get_xlim() + ax.get_ylim())
            ax.xaxis.set_ticks(range(xmi, xma, 20000))
            ax.yaxis.set_ticks(range(ymi, yma, 20000))

            ax.legend()

            ax.set_xlabel('Easting', fontsize=13, labelpad=5)
            ax.set_ylabel('Northing', fontsize=13, labelpad=5)

            # ax.xaxis.set_ticks([380600, 380700])
            # ax.yaxis.set_ticks([665400, 665500])

            # from pyhelpers.store import save_figure
            #
            # fig_pathname = "docs/source/_images/krdzg_get_trk_shp_for_pcd_tile_demo"
            # save_figure(fig, f"{fig_pathname}.svg", verbose=True)
            # save_figure(fig, f"{fig_pathname}.pdf", verbose=True)
            #
            # fig_pathname = "docs/source/_images/krdzg_get_trk_shp_for_pcd_tile_demo_zoomed_in"
            # save_figure(fig, f"{fig_pathname}.svg", verbose=True)
            # save_figure(fig, f"{fig_pathname}.pdf", verbose=True)

        .. figure:: ../_images/krdzg_get_trk_shp_for_pcd_tile_demo.*
            :name: krdzg_get_trk_shp_for_pcd_tile_demo
            :align: center
            :width: 100%

            The track shapefile with regard to the Tile (380600, 665400).

        .. figure:: ../_images/krdzg_get_trk_shp_for_pcd_tile_demo_zoomed_in.*
            :name: krdzg_get_trk_shp_for_pcd_tile_demo_zoomed_in
            :align: center
            :width: 100%

            (Zoomed-in) The track shapefile with regard to the Tile (380600, 665400).
        """

        trk_shp_ = sorted(trk_shp.geoms, key=lambda x: x.distance(tile_poly))

        if tile_poly.bounds in [(380600, 665400, 380700, 665500), (380700, 665500, 380800, 665600)]:
            shp_in_tile = [
                x.intersection(tile_poly) for x in trk_shp_[1:] if x.intersects(tile_poly)]
            trk_shp_ls = trk_shp_[1::2][:2]
        else:
            shp_in_tile = [x.intersection(tile_poly) for x in trk_shp_ if x.intersects(tile_poly)]
            trk_shp_ls = trk_shp_[:2]

        if len(shp_in_tile) > 2:
            shp_in_tile_ = sorted(
                itertools.combinations(shp_in_tile, 2), reverse=True,
                key=lambda x: x[0].distance(x[1]) + x[0].length + x[1].length)

            shp_in_tile_ = [(a, b) for a, b in shp_in_tile_ if a.distance(b) > 3]

            ls1, ls2 = sorted(shp_in_tile_[0], key=lambda x: x.length)

            buf = ls1.buffer(distance=ls1.centroid.distance(ls2) * 2, cap_style=2)
            tile_trk_shp_list = [ls1, buf.intersection(ls2)]

        elif len(shp_in_tile) == 2:
            if shp_in_tile[0].distance(shp_in_tile[1]) < 3.0:
                shp_outside_tile = [x for x in trk_shp_[2:10] if not x.intersects(tile_poly)]
                permutation = itertools.product(shp_in_tile, shp_outside_tile)
                ls_in, ls_out = max(
                    (ls for ls in permutation if ls[0].centroid.distance(ls[1]) < 4.5),
                    key=lambda x: x[0].centroid.distance(x[1]))

                tile_buf = ls_in.buffer(distance=ls_in.centroid.distance(ls_out) + 2.0, cap_style=2)
                tile_trk_shp_list = [ls_in, tile_buf.intersection(ls_out)]

            else:
                ls1, ls2 = sorted(shp_in_tile, key=lambda x: x.length)
                ls1_ = min(trk_shp_, key=lambda x: ls1.distance(x))

                buf = ls2.buffer(distance=ls1.centroid.distance(ls2) * 2, cap_style=2)
                tile_trk_shp_list = [buf.intersection(ls1_), ls2]

        else:  # len(shp_in_tile) < 2
            if len(shp_in_tile) == 1:
                trk_shp_outside_tile = [x for x in trk_shp_ls if not x.intersects(tile_poly)][0]
                dist = trk_shp_outside_tile.distance(tile_poly.exterior)
            else:
                dist = max(tile_poly.distance(x) for x in trk_shp_ls)

            buf = tile_poly.buffer(distance=dist + 2.0, cap_style=2)
            tile_trk_shp_list = list(map(lambda x: buf.intersection(x), trk_shp_ls))

        tile_trk_shp = MultiLineString(tile_trk_shp_list)

        return tile_trk_shp

    def get_tracks_shp_for_krdz_clf(self, tiles_convex_hull, tile_poly=None, **kwargs):
        """
        Get track shapefiles for clustering the KRDZ data.

        :param tiles_convex_hull: Convex hull of all tiles for the point cloud data
        :type tiles_convex_hull: Polygon
        :param tile_poly: A tile; defaults to ``None``.
        :type tile_poly: Polygon | None
        :param kwargs: [Optional] parameters of the method
            :meth:`~src.preprocessor.Track.load_tracks_shp`.
        :return: Track lines that are used for identifying KRDZ data between up and down directions.
        :rtype: MultiLineString

        **Examples**::

            >>> from src.shaft import KRDZGear
            >>> from shapely.geometry import MultiLineString
            >>> krdzg = KRDZGear()
            >>> convex_hull = krdzg.get_tiles_convex_hull(krdzg.load_tiles()['Tile_XY'])
            >>> track_shp = krdzg.get_tracks_shp_for_krdz_clf(tiles_convex_hull=convex_hull)
            >>> type(track_shp)
            shapely.geometry.multilinestring.MultiLineString
            >>> len(track_shp.geoms)
            102

        **Illustration**::

            import matplotlib.pyplot as plt
            import numpy as np
            from pyhelpers.settings import mpl_preferences

            mpl_preferences(backend='TkAgg')

            fig = plt.figure(figsize=(11, 5), constrained_layout=True)
            ax = fig.add_subplot()
            ax.set_aspect('equal', adjustable='box')

            colours = plt.get_cmap('tab10').colors

            ch_arr = np.array(convex_hull.exterior.coords)
            ax.plot(ch_arr[:, 0], ch_arr[:, 1], color=colours[1], label='Convex hull for all tiles')

            for trk_ls in track_shp.geoms:
                trk_xs, trk_xy = trk_ls.coords.xy
                ax.plot(trk_xs, trk_xy, color=colours[2])
            ax.plot([], [], color=colours[2], label='Track shapefiles for clustering the KRDZ data')

            ax.legend()

            ax.set_xlabel('Easting', fontsize=13, labelpad=5)
            ax.set_ylabel('Northing', fontsize=13, labelpad=5)

            # from pyhelpers.store import save_figure
            # fig_pathname = "docs/source/_images/krdzg_get_trk_shp_for_krdz_clf_demo"
            # save_figure(fig, f"{fig_pathname}.svg", verbose=True)
            # save_figure(fig, f"{fig_pathname}.pdf", verbose=True)

        .. figure:: ../_images/krdzg_get_trk_shp_for_krdz_clf_demo.*
            :name: krdzg_get_trk_shp_for_krdz_clf_demo
            :align: center
            :width: 100%

            The track shapefile for clustering the KRDZ data.
        """

        if self.db_instance is None:
            self.db_instance = TrackFixityDB()
        self.trk.db_instance = self.db_instance

        trk_shp_data = self.trk.load_tracks_shp(
            elr=['ECM7', 'ECM8'] if self.elr == 'ECM8' else self.elr, **kwargs)

        trk_shp_dat = trk_shp_data[
            trk_shp_data.geometry.map(lambda x: x.intersects(tiles_convex_hull))]

        trk_shp = MultiLineString(trk_shp_dat.geometry.to_list())
        if tile_poly is not None:
            trk_shp = self.get_pcd_tile_tracks_shp(trk_shp=trk_shp, tile_poly=tile_poly)

        return trk_shp

    def get_tracks_shp_reference(self, trk_shp, tile_poly):
        """
        Get track shapefiles outside a buffer of a given tile (for point cloud data),
        used as a reference for clustering the KRDZ data within the tile.

        :param trk_shp: A number of geometry objects of track lines.
        :type trk_shp: MultiLineString
        :param tile_poly: A tile.
        :type tile_poly: Polygon
        :return: Track lines data used as reference for clustering the KRDZ data.
        :rtype: tuple

        **Examples**::

            >>> from src.shaft import KRDZGear
            >>> krdzg = KRDZGear()
            >>> trk_shp_data = krdzg.trk.load_tracks_shp(elr=krdzg.elr)
            >>> track_shp = MultiLineString(trk_shp_data.geometry.to_list())
            >>> type(track_shp)
            shapely.geometry.multilinestring.MultiLineString
            >>> len(track_shp.geoms)
            440
            >>> tile_xy = (380600, 665400)
            >>> tile_polygon = krdzg.get_pcd_tile_polygon(tile_xy=tile_xy)
            >>> type(tile_polygon)
            shapely.geometry.polygon.Polygon
            >>> print(tile_polygon.bounds)
            (380600.0, 665400.0, 380700.0, 665500.0)
            >>> tile_tracks_shp, tile_tracks_shp_ref = krdzg.get_tracks_shp_reference(
            ...     trk_shp=track_shp, tile_poly=tile_polygon)
            >>> type(tile_tracks_shp)
            shapely.geometry.multilinestring.MultiLineString
            >>> len(tile_tracks_shp.geoms)
            2
            >>> type(tile_tracks_shp_ref)
            shapely.geometry.multilinestring.MultiLineString
            >>> len(tile_tracks_shp_ref.geoms)
            442
        """

        tile_trk_shp = self.get_pcd_tile_tracks_shp(trk_shp=trk_shp, tile_poly=tile_poly)
        tile_trk_shp_ = max(tile_trk_shp.geoms, key=lambda x: x.length)

        tile_trk_buf = tile_trk_shp_.buffer(
            distance=tile_trk_shp_.hausdorff_distance(tile_poly.exterior), cap_style=3)

        tile_trk_shp_ref = trk_shp.difference(tile_trk_buf)

        return tile_trk_shp, tile_trk_shp_ref

    @staticmethod
    def _get_extra_ref_lines(tiles_envelope, xyz1_geom_, xyz2_geom_):
        # # Calculate (linear function) slope
        # lf_slopes = list(map(functools.partial(calculate_slope, na_val=0), (xyz1, xyz2)))

        env_ll, env_lr, env_ur, env_ul, _ = map(Point, tiles_envelope.boundary.coords)

        temp = LineString(np.unique(np.array(sorted(xyz2_geom_.coords)), axis=0))
        temp = LineString(temp.boundary.geoms)
        dist = temp.centroid.hausdorff_distance(tiles_envelope.exterior)

        # Reference line 1
        par_side = 'right'  # if any(s > 0 for s in lf_slopes) else 'left'
        # distance=xyz2_geom_.centroid.distance(ref_line_par) if d is None else d
        # resolution=len(xyz2_geom_.coords)
        ref_l1 = temp.parallel_offset(distance=dist, side=par_side)

        cen_in_between = LineString([xyz1_geom_.centroid, xyz2_geom_.centroid]).centroid

        # Reference line 2
        par_side = 'left'
        ref_l2 = temp.parallel_offset(distance=dist, side=par_side)
        ref_l2 = LineString([cen_in_between.centroid, ref_l2.centroid])

        # Reference line 3
        if calculate_slope(np.array(temp.coords)) > 0:
            side_ends_3 = [env_ll, env_ul]
        else:
            side_ends_3 = [env_lr, env_ur]
        ref_p3 = point_projected_to_line(temp.centroid, LineString(side_ends_3))[1]
        ref_l3 = LineString([ref_p3, cen_in_between.centroid])

        # Reference line 4
        if calculate_slope(np.array(temp.coords)) > 0:
            side_ends_4 = [env_ul, env_ur]
        else:
            side_ends_4 = [env_ll, env_lr]
        ref_p4 = point_projected_to_line(temp.centroid, LineString(side_ends_4))[1]
        ref_l4 = LineString([ref_p4, cen_in_between.centroid])

        return ref_l1, ref_l2, ref_l3, ref_l4

    def distinguish_between_up_and_down(self, xyz1, xyz2, tiles_convex_hull, ref_line_par,
                                        tile_poly, tile_trk_shp_ref):
        # noinspection PyShadowingNames
        """
        Distinguish the (up and down) running directions of the two given arrays of rail track data.

        The rail track data could be KRDZ data and track shapefile data.

        :param xyz1: One array of rail track data.
        :type xyz1: numpy.ndarray
        :param xyz2: Another array of rail track data.
        :type xyz2: numpy.ndarray
        :param tiles_convex_hull: Convex hull of all tiles for the point cloud data.
        :type tiles_convex_hull: Polygon
        :param ref_line_par: A reference line that is parallel to a pre-specified one.
        :type ref_line_par: LineString
        :param tile_poly: A tile.
        :type tile_poly: Polygon
        :param tile_trk_shp_ref: A subsection of track shapefile used for reference.
        :type tile_trk_shp_ref: LineString
        :return: Data of rail track in the up direction and that in the down direction.
        :rtype: tuple

        **Examples**::

            >>> from src.shaft import KRDZGear
            >>> from src.utils import get_tile_xy
            >>> from sklearn.cluster import DBSCAN
            >>> from pyhelpers.geom import find_shortest_path
            >>> import numpy as np
            >>> krdzg = KRDZGear()
            >>> pcd_date, tile_xy = '202004', (399600, 654100)
            >>> tile_x, tile_y = get_tile_xy(tile_xy=tile_xy)
            >>> # KRDZ data
            >>> krdz_xyz = krdzg.load_pcd_krdz(tile_xy=(tile_x, tile_y), pcd_date=pcd_date)
            >>> dat = krdz_xyz['LeftTopOfRail']
            >>> clusters = DBSCAN(eps=2, min_samples=2, algorithm='brute').fit(dat)
            >>> labels = np.unique(clusters.labels_)
            >>> xyz_1 = find_shortest_path(dat[clusters.labels_ == labels[0]])
            >>> xyz_2 = find_shortest_path(dat[clusters.labels_ != labels[0]])
            >>> len(xyz_1)
            100
            >>> len(xyz_2)
            100
            >>> ch, ref_lp, trks = krdzg.get_reference_objects_for_krdz_clf(pcd_date, par_dist=5000)
            >>> tile_polygon = krdzg.get_pcd_tile_polygon(tile_xy=(tile_x, tile_y), as_geom=True)
            >>> _, tile_tracks_shp_ref = krdzg.get_tracks_shp_reference(trks, tile_polygon)
            >>> up_xyz, down_xyz = krdzg.distinguish_between_up_and_down(
            ...     xyz1=xyz_1, xyz2=xyz_2, tiles_convex_hull=ch, ref_line_par=ref_lp,
            ...     tile_poly=tile_polygon, tile_trk_shp_ref=tile_tracks_shp_ref)
            >>> type(up_xyz)
            shapely.geometry.linestring.LineString
            >>> print(up_xyz.wkt)
            LINESTRING Z (399641.403 654100.317 37.556, 399641.445 654101.316 37.558, 399641.48...
            >>> type(down_xyz)
            shapely.geometry.linestring.LineString
            >>> print(down_xyz.wkt)
            LINESTRING Z (399645.168 654199.431 37.987, 399645.203 654198.433 37.983, 399645.23...

        .. seealso::

            For detailed illustration of the method, see `debugging.ipynb
            <https://github.com/mikeqfu/network-rail-track-fixity-layer/
            blob/master/src/demos/debugging.ipynb>`_.
        """

        # Convert the array data to geometry object
        xyz_geoms = map(
            lambda x: LineString(np.vstack([x, x]) if x.shape == (1, 3) else x), (xyz1, xyz2))
        xyz1_geom, xyz2_geom = sorted(xyz_geoms, key=lambda x: x.length)

        # Offset
        xyz1_geom_, xyz2_geom_ = offset_ls(ls1=xyz1_geom, ls2=xyz2_geom)

        tiles_envelope = tiles_convex_hull.envelope

        # `ref_line_1` - checking the relative position of xyz1 and xyz2
        # `ref_line_2`, `ref_line_3` and `ref_line_4` - checking if the section is a 'reversed curve'
        ref_line_1, ref_line_2, ref_line_3, ref_line_4 = self._get_extra_ref_lines(
            tiles_envelope, xyz1_geom_, xyz2_geom_)

        cond = xyz1_geom_.distance(ref_line_1) < xyz2_geom_.distance(ref_line_1)

        xyz1_cen, xyz2_cen, env_cen = (
            xyz1_geom_.centroid, xyz2_geom_.centroid, tiles_envelope.centroid)
        cond1 = xyz1_cen.distance(env_cen) < xyz2_cen.distance(env_cen)

        ref_line_par_adj = self.get_adjusted_ref_line_par(
            ref_line_par=ref_line_par, tile_poly=tile_poly, aux_ref=ref_line_1, centroid=True)
        cond2 = xyz1_cen.distance(ref_line_par_adj) < xyz2_cen.distance(ref_line_par_adj)

        # ref_vtx = shapely.ops.nearest_points(ref_ls_par_adj, MultiPoint(tile_poly.exterior.coords))[1]
        # cond3 = xyz1_cen.distance(ref_vtx) < xyz2_cen.distance(ref_vtx)

        cond3 = xyz1_cen.distance(ref_line_par.centroid) < xyz2_cen.distance(ref_line_par.centroid)
        cond3_ = xyz1_cen.distance(ref_line_par) < xyz2_cen.distance(ref_line_par)

        cond4, cond5, cond6 = map(
            lambda x: x.intersects(tile_trk_shp_ref), (ref_line_2, ref_line_3, ref_line_4))

        if cond:  # xyz1_geom_.distance(ref_line_1) < xyz2_geom_.distance(ref_line_1)
            cond5 = False if sum([cond1, cond2]) == 1 else cond5
            if (cond4 and cond5) or cond6:
                if (cond1 and cond2) or (cond2 and cond3):
                    up_dat, down_dat = xyz1_geom, xyz2_geom
                else:
                    up_dat, down_dat = xyz2_geom, xyz1_geom
            else:
                if sum([cond1, cond2, cond3, cond3_]) >= 2:  # all([cond1, cond2, cond3])
                    up_dat, down_dat = xyz1_geom, xyz2_geom
                else:
                    if cond2 and not any([cond1, cond3, cond3_]):
                        up_dat, down_dat = xyz1_geom, xyz2_geom
                    else:
                        up_dat, down_dat = xyz2_geom, xyz1_geom

        else:  # xyz1_geom_.distance(ref_line_1) > xyz2_geom_.distance(ref_line_1)
            cond5 = False if sum([cond1, cond2, cond3]) == 2 else cond5
            if (cond4 and cond5) or cond6:
                if (cond1 and cond2) or (cond1 and cond3):  # xyz1_geom -above- xyz2_geom
                    up_dat, down_dat = xyz1_geom, xyz2_geom
                else:
                    up_dat, down_dat = xyz2_geom, xyz1_geom
            else:
                if cond1 or sum([cond2, cond3, cond3_]) == 1:  # any([cond1, cond2, cond3])
                    if cond1 and cond3:
                        ref11 = point_projected_to_line(ref_line_par_adj, ref_line_3)[1]
                        cond11 = xyz1_cen.distance(ref11) < xyz2_cen.distance(ref11)
                        ref12 = point_projected_to_line(ref_line_par_adj, ref_line_4)[1]
                        cond12 = xyz1_cen.distance(ref12) < xyz2_cen.distance(ref12)
                        if all([cond11, cond12]):
                            up_dat, down_dat = xyz1_geom, xyz2_geom
                        else:
                            up_dat, down_dat = xyz2_geom, xyz1_geom
                    else:
                        up_dat, down_dat = xyz2_geom, xyz1_geom
                else:
                    if sum([cond1, cond2, cond3, cond3_]) == 0:
                        up_dat, down_dat = xyz2_geom, xyz1_geom
                    else:
                        up_dat, down_dat = xyz1_geom, xyz2_geom

        return up_dat, down_dat

    def get_reference_objects_for_krdz_clf(self, pcd_date, par_dist=5000, ret_tile_names=False):
        """
        Get three different reference objects
        for distinguishing the KRDZ data between up and down directions.

        :param pcd_date: Date of point cloud data.
        :type pcd_date: str | int
        :param par_dist: Distance to parallel offset the key reference line; defaults to ``5000``.
        :type par_dist: int | float
        :param ret_tile_names: Whether to return tile names as well; defaults to ``False``.
        :type ret_tile_names: bool
        :return: Reference objects for distinguishing the KRDZ data between up and down directions.
        :rtype: tuple

        **Examples**::

            >>> from src.shaft import KRDZGear
            >>> krdzg = KRDZGear()
            >>> ref_objects = krdzg.get_reference_objects_for_krdz_clf(pcd_date='202004')
            >>> convex_hull, ref_line_paralleled, track_shp = ref_objects
            >>> type(convex_hull)
            shapely.geometry.polygon.Polygon
            >>> type(ref_line_paralleled)
            shapely.geometry.linestring.LineString
            >>> type(track_shp)
            shapely.geometry.multilinestring.MultiLineString

        **Illustration**::

            import matplotlib.pyplot as plt
            from pyhelpers.settings import mpl_preferences

            mpl_preferences(backend='TkAgg', font_name='Times New Roman')

            fig = plt.figure(figsize=(11, 5), constrained_layout=True)
            ax = fig.add_subplot()
            ax.set_aspect('equal', adjustable='box')

            colours = plt.get_cmap('tab10').colors

            xs, ys = convex_hull.exterior.coords.xy
            ax.plot(xs, ys, color=colours[1], label='Convex hull for all tiles')

            xs, ys = ref_line_paralleled.coords.xy
            ax.plot(xs, ys, color=colours[0], label='Ref line (paralleled)')

            for trk_ls in track_shp.geoms:
                xs, ys = trk_ls.coords.xy
                ax.plot(xs, ys, color=colours[2])
            ax.plot([], [], color=colours[2], label='Track shapefiles')

            ax.legend()

            ax.set_xlabel('Easting', fontsize=13, labelpad=5)
            ax.set_ylabel('Northing', fontsize=13, labelpad=5)

            # from pyhelpers.store import save_figure
            # fig_pathname = "docs/source/_images/krdzg_get_ref_obj_for_krdz_clf_demo"
            # save_figure(fig, f"{fig_pathname}.svg", verbose=True)
            # save_figure(fig, f"{fig_pathname}.pdf", verbose=True)

        .. figure:: ../_images/krdzg_get_ref_obj_for_krdz_clf_demo.*
            :name: krdzg_get_ref_obj_for_krdz_clf_demo
            :align: center
            :width: 100%

            Main reference objects for clustering the KRDZ data.
        """

        pcd_tiles_meta = self.load_tiles(pcd_date=pcd_date)
        pcd_tiles, tile_names = pcd_tiles_meta.Tile_XY, pcd_tiles_meta.Tile_Name
        tiles_convex_hull = self.get_tiles_convex_hull(pcd_tiles=pcd_tiles)

        ref_line = self.get_key_reference(tiles_convex_hull=tiles_convex_hull)
        ref_line_coords = np.array(ref_line.coords)
        ref_line_par = ref_line.parallel_offset(
            distance=par_dist, side='right' if calculate_slope(ref_line_coords) > 0 else 'left')

        trk_shp = self.get_tracks_shp_for_krdz_clf(tiles_convex_hull=tiles_convex_hull)

        ref_location = tiles_convex_hull, ref_line_par, trk_shp

        if ret_tile_names:
            ref_location = ref_location, tile_names

        return ref_location

    @staticmethod
    def _get_aux_trk_shp(trk_shp_ls, xyz_geom):
        if xyz_geom.length < trk_shp_ls.length:
            aux = np.array(offset_ls(ls1=xyz_geom, ls2=trk_shp_ls)[1].coords)
        else:
            aux = np.array(trk_shp_ls.coords)

        # aux = find_the_shortest_path(aux[aux[:, 0].argsort()])
        aux = find_shortest_path(points_sequence=aux)

        return aux

    @staticmethod
    def _swap_left_right_keys(down_keys):
        down_keys_ = []

        for dk in down_keys:
            temp1 = re.sub(r'Left', '%left_temp%', dk)
            # replace 'Left' with 'Right' in temp1
            temp2 = re.sub(r'Right', 'Left', temp1)
            # replace '%left_temp%' with 'Right' in temp2
            dk_ = re.sub('%left_temp%', 'Right', temp2)
            down_keys_.append(dk_)

        return down_keys_

    def classify_krdz(self, tile_xy, pcd_date, ref_objects=None, update=False, verbose=True,
                      **kwargs):
        """
        Classify KRDZ data by railway directions.

        :param tile_xy: X and Y coordinates in reference to a tile for the point cloud data.
        :type tile_xy: tuple | list | str
        :param pcd_date: Date of the point cloud data.
        :type pcd_date: str | int
        :param ref_objects: Reference objects for identifying the running direction.
        :type ref_objects: tuple | None
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of `sklearn.cluster.DBSCAN`_.
        :return: Coordinates of upside and downside rail heads, and associated labels.
        :rtype: dict

        .. _`sklearn.cluster.DBSCAN`:
            https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

        **Examples**::

            >>> from src.shaft import KRDZGear
            >>> krdzg = KRDZGear()
            >>> # tile_xy, pcd_date = (340500, 674200), '201910'
            >>> classified_krdz_201910 = krdzg.classify_krdz(
            ...     tile_xy=(340500, 674200), pcd_date='201910', n_jobs=-1)
            >>> type(classified_krdz_201910)
            dict
            >>> len(classified_krdz_201910)
            10
            >>> list(classified_krdz_201910.keys())
            ['Up_LeftTopOfRail',
             'Up_LeftRunningEdge',
             'Up_RightTopOfRail',
             'Up_RightRunningEdge',
             'Up_Centre',
             'Down_RightTopOfRail',
             'Down_RightRunningEdge',
             'Down_LeftTopOfRail',
             'Down_LeftRunningEdge',
             'Down_Centre']
            >>> ult_ls_201910 = classified_krdz_201910['Up_LeftTopOfRail']
            >>> type(ult_ls_201910)
            shapely.geometry.linestring.LineString
            >>> ult_ls_201910.length
            105.9959864289052
            >>> urt_ls_201910 = classified_krdz_201910['Up_RightTopOfRail']
            >>> urt_ls_201910.length
            105.99667584951499
            >>> # tile_xy, pcd_date = (340500, 674200), '202004'
            >>> classified_krdz_202004 = krdzg.classify_krdz(
            ...     tile_xy=(340500, 674200), pcd_date='202004', n_jobs=-1)
            >>> ult_ls_202004 = classified_krdz_202004['Up_LeftTopOfRail']
            >>> ult_ls_202004.length
            105.99856695815154

        .. seealso::

            For detailed illustration of the method, see `debugging.ipynb
            <https://github.com/mikeqfu/network-rail-track-fixity-layer/
            blob/master/src/demos/debugging.ipynb>`_.
        """

        tile_x, tile_y = get_tile_xy(tile_xy=tile_xy)

        path_to_pickle = cd(self.krdz_dir_(pcd_date), f"Tile_X+0000{tile_x}_Y+0000{tile_y}.pickle")

        if os.path.isfile(path_to_pickle) and not update:
            classified_data = load_data(path_to_pickle)

        else:
            krdz_xyz = self.load_pcd_krdz(tile_xy=(tile_x, tile_y), pcd_date=pcd_date)
            tile_poly = self.get_pcd_tile_polygon(tile_xy=(tile_x, tile_y), as_geom=True)

            if ref_objects is None:
                tiles_convex_hull, ref_line_par, trk_shp = self.get_reference_objects_for_krdz_clf(
                    pcd_date=pcd_date, par_dist=5000)
            else:
                tiles_convex_hull, ref_line_par, trk_shp = ref_objects

            empty_geom = LineString([])

            up_rail_head, down_rail_head = [], []
            for cat, dat in krdz_xyz.items():
                # Loop through the raw elements mixed with the data of both up and down directions

                if dat.size > 0:  # if there is data within the given tile
                    if len(dat) > 1:  # Create a clustering model
                        clusters = DBSCAN(eps=2, min_samples=2, algorithm='brute', **kwargs).fit(dat)
                        labels, idx = np.unique(clusters.labels_, return_index=True)
                        xyz1 = find_shortest_path(dat[clusters.labels_ == labels[0]])
                        xyz2 = find_shortest_path(dat[clusters.labels_ != labels[0]])[::-1]
                    else:
                        xyz1, xyz2 = find_shortest_path(dat), np.array([], dtype=dat.dtype)

                    tile_trk_shp, tile_trk_shp_ref = self.get_tracks_shp_reference(
                        trk_shp=trk_shp, tile_poly=tile_poly)

                    if any(a.size == 0 for a in (xyz1, xyz2)):
                        xyz = max((xyz1, xyz2), key=len)

                        xyz_geom = LineString(np.vstack([xyz, xyz]) if xyz.shape == (1, 3) else xyz)
                        xyz_midpt = shapely.ops.nearest_points(
                            xyz_geom.centroid, MultiPoint(xyz_geom.coords))[1]

                        # aux1, aux2 = map(get_aux_trk_shp, tile_trk_shp.geoms)
                        aux1, aux2 = map(
                            functools.partial(self._get_aux_trk_shp, xyz_geom=xyz_geom),
                            tile_trk_shp.geoms)

                        up_aux, down_aux = self.distinguish_between_up_and_down(
                            xyz1=aux1, xyz2=aux2, tiles_convex_hull=tiles_convex_hull,
                            ref_line_par=ref_line_par, tile_poly=tile_poly,
                            tile_trk_shp_ref=tile_trk_shp_ref)

                        if xyz_midpt.distance(up_aux) < xyz_midpt.distance(down_aux):
                            up_dat, down_dat = xyz_geom, empty_geom
                        else:
                            up_dat, down_dat = empty_geom, xyz_geom

                    else:
                        up_dat, down_dat = self.distinguish_between_up_and_down(
                            xyz1=xyz1, xyz2=xyz2, tiles_convex_hull=tiles_convex_hull,
                            ref_line_par=ref_line_par, tile_poly=tile_poly,
                            tile_trk_shp_ref=tile_trk_shp_ref)

                else:
                    up_dat, down_dat = empty_geom, empty_geom

                up_rail_head.append(up_dat)
                down_rail_head.append(down_dat)

                gc.collect()

            classified_data = dict(zip(['Up_' + k for k in krdz_xyz.keys()], up_rail_head))

            temp = dict(zip(
                ['Down_' + k for k in self._swap_left_right_keys(krdz_xyz.keys())], down_rail_head))
            classified_data.update(temp)

            save_data(classified_data, path_to_pickle, verbose=verbose)

        return classified_data

    def view_classified_krdz(self, tile_xy, pcd_date, projection='3d', cmap_name='tab10',
                             add_title=False, save_as=None, dpi=600, verbose=False, **kwargs):
        # noinspection PyShadowingNames
        """
        View classified KRDZ data of the rail heads of point cloud data.

        :param tile_xy: X and Y coordinates in reference to a tile for the point cloud data.
        :type tile_xy: tuple | list | str
        :param pcd_date: Date of the point cloud data.
        :type pcd_date: str | int
        :param projection: Projection type of the subplot; defaults to ``'3d'``.
        :type projection: str | None
        :param cmap_name: Name of a matplotlib color map; defaults to ``'tab10'``.
        :type cmap_name: str | None
        :param add_title: Whether to add a title to the plot; defaults to ``False``.
        :type add_title: bool
        :param save_as: File format that the view is saved as; defaults to ``None``.
        :type save_as: str | None
        :param dpi: DPI for saving image; defaults to ``600``.
        :type dpi: int | None
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters of the method
            :meth:`~src.shaft.krdz_gear.KRDZGear.classify_krdz`.

        **Examples**::

            >>> from src.shaft import KRDZGear
            >>> from pyhelpers.settings import mpl_preferences
            >>> mpl_preferences(backend='TkAgg')
            >>> krdzg = KRDZGear()
            >>> tile_xy = (340500, 674200)
            >>> # 3D view
            >>> # krdzg.view_classified_krdz(tile_xy, '201910', save_as=".svg", verbose=True)
            >>> krdzg.view_classified_krdz(tile_xy=tile_xy, pcd_date='201910')

        .. figure:: ../_images/krdzg_view_classified_krdz_x340500y674200_201910_3d.*
            :name: krdzg_view_classified_krdz_x340500y674200_201910_3d
            :align: center
            :width: 88%

            A 3D view of the classified KRDZ data in Tile (340500, 674200) in October 2019.

        .. code-block:: python

            >>> # 2D plot - Vertical view
            >>> # krdzg.view_classified_krdz(tile_xy, '201910', None, save_as=".svg", verbose=True)
            >>> krdzg.view_classified_krdz(tile_xy=tile_xy, pcd_date='201910', projection=None)

        .. figure:: ../_images/krdzg_view_classified_krdz_x340500y674200_201910.*
            :name: krdzg_view_classified_krdz_x340500y674200_201910
            :align: center
            :width: 100%

            A vertical view of the classified KRDZ data in Tile (340500, 674200) in October 2019.

        .. code-block:: python

            >>> # 3D view
            >>> # krdzg.view_classified_krdz(tile_xy, '202004', save_as=".svg", verbose=True)
            >>> krdzg.view_classified_krdz(tile_xy=tile_xy, pcd_date='202004')

        .. figure:: ../_images/krdzg_view_classified_krdz_x340500y674200_202004_3d.*
            :name: krdzg_view_classified_krdz_x340500y674200_202004_3d
            :align: center
            :width: 88%

            A 3D view of the classified KRDZ data in Tile (340500, 674200) in April 2020.

        .. code-block:: python

            >>> # 2D plot - Vertical view
            >>> # krdzg.view_classified_krdz(tile_xy, '202004', None, save_as=".svg", verbose=True)
            >>> krdzg.view_classified_krdz(tile_xy=tile_xy, pcd_date='202004', projection=None)

        .. figure:: ../_images/krdzg_view_classified_krdz_x340500y674200_202004.*
            :name: krdzg_view_classified_krdz_x340500y674200_202004
            :align: center
            :width: 100%

            A vertical view of the classified KRDZ data in Tile (340500, 674200) in April 2020.
        """

        clf_rail_head = self.classify_krdz(tile_xy=tile_xy, pcd_date=pcd_date, **kwargs)

        if not all(dat.is_empty for dat in clf_rail_head.values()):
            colours = plt.colormaps.get_cmap(cmap_name).colors

            line_widths = [
                1 if 'Edge' in label else (3 if 'Top' in label else 2)
                for label in clf_rail_head.keys()]

            rail_head_xyz_data = zip(clf_rail_head.items(), colours[:10], line_widths)

            if projection:
                fig = plt.figure(figsize=(9, 8))
                ax = fig.add_subplot(projection=projection)
                ax.tick_params(axis='z', which='major', pad=12)  # , labelsize=18
                for (label, xyz), colour, lw in rail_head_xyz_data:
                    if not xyz.is_empty:
                        xyz_ = np.asarray(xyz.coords)
                        ax.plot3D(
                            xyz_[:, 0], xyz_[:, 1], xyz_[:, 2], color=colour, label=label,
                            linewidth=lw)
                ax.set_xlabel('Easting', fontsize=18, labelpad=12)
                ax.set_ylabel('Northing', fontsize=18, labelpad=14)
                ax.set_zlabel('Elevation', fontsize=18, labelpad=22)

            else:
                fig = plt.figure(figsize=(11, 5))
                ax = fig.add_subplot(aspect='equal', adjustable='box')
                for (label, xyz), colour, lw in rail_head_xyz_data:
                    if not xyz.is_empty:
                        xyz_ = np.asarray(xyz.coords)
                        ax.plot(xyz_[:, 0], xyz_[:, 1], color=colour, label=label, linewidth=lw)
                ax.set_xlabel('Easting', fontsize=18, labelpad=8)
                ax.set_ylabel('Northing', fontsize=18, labelpad=8)
                ax.grid()

            ax.tick_params(axis='both', which='major')  # , labelsize=18
            ax.ticklabel_format(useOffset=False)

            tile_x, tile_y = get_tile_xy(tile_xy=tile_xy)

            if add_title:
                title_date = datetime.datetime.strptime(str(pcd_date), '%Y%m').strftime('%B %Y')
                ax.set_title(f"Classified KRDZ data within Tile{tile_x, tile_y} in {title_date}.")

            ax.legend(loc='best', numpoints=1, ncol=1, fontsize=15)

            fig.tight_layout()
            if projection:
                fig.subplots_adjust(left=0, bottom=0, right=0.85, top=1)

            if save_as:
                suffix = f"_{projection}" if projection else ""
                fig_filename = f"krdzg_view_classified_krdz_x{tile_x}y{tile_y}_{pcd_date}{suffix}"
                for save_as_ in {save_as, ".svg", ".pdf"}:
                    path_to_fig = cd_docs_source("_images", fig_filename + save_as_)
                    save_figure(fig, path_to_fig, dpi=dpi, transparent=True, verbose=verbose)

    def gather_classified_krdz(self, pcd_date, set_index=None, update=False, verbose=False,
                               **kwargs):
        """
        Collect together all classified KRDZ data for a given date.

        :param pcd_date: Date of the point cloud data.
        :type pcd_date: str | int
        :param set_index: Whether to set an index (or indexes),
            or what to be set as an index (or indexes); defaults to ``None``.
        :type set_index: bool | list | None
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``True``.
        :type verbose: bool | int
        :return: KRDZ rail head data (classified by up and down directions).
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft import KRDZGear
            >>> krdzg = KRDZGear()
            >>> # -- October 2019 -----------------------------------
            >>> # classified_krdz_201910 = tm.gather_classified_krdz(
            ... #    pcd_date='201910', update=True, verbose=True, n_jobs=-1)
            >>> classified_krdz_201910 = krdzg.gather_classified_krdz(pcd_date='201910')
            >>> classified_krdz_201910
                  Year  ...                                           geometry
            0     2019  ...  LINESTRING Z (340199.193 674110.113 33.165, 34...
            1     2019  ...  LINESTRING Z (340199.18 674110.147 33.15, 3401...
            2     2019  ...  LINESTRING Z (340199.601 674111.877 33.163, 34...
            3     2019  ...  LINESTRING Z (340199.613 674111.843 33.148, 34...
            4     2019  ...  LINESTRING Z (340199.866 674111.17 33.163, 340...
            ...    ...  ...                                                ...
            9807  2019  ...  LINESTRING Z (340708.001 674299.684 32.27, 340...
            9808  2019  ...  LINESTRING Z (340707.987 674299.719 32.253, 34...
            9809  2019  ...  LINESTRING Z (340703.726 674299.666 32.214, 34...
            9810  2019  ...  LINESTRING Z (340704.674 674299.985 32.201, 34...
            9811  2019  ...  LINESTRING Z (340705.871 674299.677 32.242, 34...
            [9812 rows x 7 columns]
            >>> # -- April 2020 -------------------------------------
            >>> # classified_krdz_202004 = tm.gather_classified_krdz(
            ... #    pcd_date='202004', update=True, verbose=True, n_jobs=-1)
            >>> classified_krdz_202004 = krdzg.gather_classified_krdz(pcd_date='202004')
            >>> classified_krdz_202004
                  Year  ...                                           geometry
            0     2020  ...  LINESTRING Z (340132.744 674085.354 33.344, 34...
            1     2020  ...  LINESTRING Z (340132.731 674085.389 33.329, 34...
            2     2020  ...  LINESTRING Z (340132.196 674086.761 33.347, 34...
            3     2020  ...  LINESTRING Z (340132.21 674086.727 33.332, 340...
            4     2020  ...  LINESTRING Z (340132.47 674086.058 33.345, 340...
            ...    ...  ...                                                ...
            9812  2020  ...  LINESTRING Z (399600.071 654486.99 39.58, 3996...
            9813  2020  ...  LINESTRING Z (399600.106 654486.999 39.567, 39...
            9814  2020  ...  LINESTRING Z (399600.129 654492.216 39.682, 39...
            9815  2020  ...  LINESTRING Z (399600.096 654492.205 39.664, 39...
            9816  2020  ...  LINESTRING Z (399600.241 654489.12 39.628, 399...
            [9817 rows x 7 columns]
        """

        ref_objects, tile_names = self.get_reference_objects_for_krdz_clf(
            pcd_date=pcd_date, ret_tile_names=True)

        # pcd_tiles_meta.Tile_XY.map(lambda x: shapely.wkt.loads(x).exterior.coords[0])
        pcd_tiles_coords = tile_names.map(get_tile_xy)

        if verbose:
            pcd_date_ = datetime.datetime.strptime(str(pcd_date), '%Y%m').strftime('%B %Y')
            print(f"Processing KRDZ data of {pcd_date_} ... ")

        dat, total_no, current_no = [], len(pcd_tiles_coords), 1

        for (tile_x, tile_y), tile_name in zip(pcd_tiles_coords, tile_names):
            if verbose:
                print(f"\t{current_no}/{total_no}: {tile_name}", end=" ... ")

            try:
                classified_rail_head = self.classify_krdz(
                    tile_xy=(tile_x, tile_y), pcd_date=pcd_date, ref_objects=ref_objects,
                    update=update, verbose=False, **kwargs)

                for label, geom in classified_rail_head.items():
                    if not geom.is_empty:
                        direction, element = label.split('_')
                        dat.append([tile_x, tile_y, direction, element, geom])

                if verbose:
                    print("Done.")

            except Exception as e:
                if verbose:
                    print(f"Failed. {e}.")

                break

            current_no += 1

        classified_krdz_data = pd.DataFrame(
            data=dat, columns=['Tile_X', 'Tile_Y', 'Direction', 'Element', 'geometry'])
        year, month = data_date_to_year_month(pcd_date)
        classified_krdz_data.insert(0, 'Year', year)
        classified_krdz_data.insert(1, 'Month', month)

        if set_index is not None and set_index is not False:
            if isinstance(set_index, list):
                keys = set_index.copy()
            else:  # set_index is True:
                keys = ['Year', 'Month', 'Tile_X', 'Tile_Y', 'Direction']
            classified_krdz_data.set_index(keys=keys, inplace=True)

        return classified_krdz_data

    def import_classified_krdz(self, update=False, verbose=True, **kwargs):
        """
        Import the classified KRDZ rail head data into the project database.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of `pyhelpers.dbms.PostgreSQL.import_data`_.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.shaft import KRDZGear
            >>> krdzg = KRDZGear()
            >>> krdzg.import_classified_krdz(if_exists='replace', verbose=True)
            To import classified rail head data into the table "PCD"."KRDZ_Classified"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/pcd_krdz_classified_tbl.*
            :name: pcd_krdz_classified_tbl
            :align: center
            :width: 100%

            Snapshot of the "PCD"."KRDZ_Classified" table.
        """

        schema_name, table_name = self.SCHEMA_NAME, self.KRDZ_CL_TABLE_NAME
        tbl_name = f'"{schema_name}"."{table_name}"'

        if confirmed("To import classified rail head data into the table {}?\n".format(tbl_name)):
            classified_rail_head_data = []

            for pcd_date in self.data_dates:
                classified_rail_head_dat = self.gather_classified_krdz(
                    pcd_date=pcd_date, update=update, verbose=True if verbose == 2 else False)

                classified_rail_head_data.append(classified_rail_head_dat)

            classified_rail_head_data = pd.concat(
                classified_rail_head_data, axis=0, ignore_index=True)

            if verbose:
                print("Importing the data", end=" ... ")

            try:
                self.db_instance.import_data(
                    data=classified_rail_head_data, schema_name=schema_name, table_name=table_name,
                    method=self.db_instance.psql_insert_copy, confirmation_required=False,
                    **kwargs)

                print("Done.") if verbose else ""

            except Exception as e:
                print(f"Failed. {e}.")

    def load_classified_krdz(self, tile_xy=None, pcd_date=None, direction=None, element=None,
                             as_dict=False):
        """
        Load data of classified KRDZ rail head data from the project database.

        :param tile_xy: Easting and northing of a tile for the point cloud data;
            defaults to ``None``.
        :type tile_xy: tuple | list | str | None
        :param pcd_date: Date of the point cloud data; defaults to ``None``.
        :type pcd_date: str | int | None
        :param direction: Railway direction;
            when ``direction=None`` (default), it refers to both up and down directions.
        :type direction: str | None
        :param element: Element of rail head;
            when ``element=None`` (default), it refers to all available elements.
        :type element: str | list | None
        :param as_dict: Whether to convert the retrieved dataframe to dict format;
            defaults to ``False``
        :type as_dict: bool
        :return: KRDZ rail head data (classified by up and down directions).
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft import KRDZGear
            >>> krdzg = KRDZGear()
            >>> krdz_dat_201910 = krdzg.load_classified_krdz(pcd_date='201910')
            >>> krdz_dat_201910.head()
               Year  ...                                           geometry
            0  2019  ...  LINESTRING Z (340199.193 674110.113 33.165, 34...
            1  2019  ...  LINESTRING Z (340199.18 674110.147 33.15, 3401...
            2  2019  ...  LINESTRING Z (340199.601 674111.877 33.163, 34...
            3  2019  ...  LINESTRING Z (340199.613 674111.843 33.148, 34...
            4  2019  ...  LINESTRING Z (340199.866 674111.17 33.163, 340...
            [5 rows x 7 columns]
            >>> krdz_dat_201910.shape
            (9812, 7)
            >>> krdz_x340500y674200_201910 = krdzg.load_classified_krdz(
            ...     tile_xy=(340500, 674200), pcd_date='201910')
            >>> krdz_x340500y674200_201910.shape
            (10, 7)
            >>> krdz_x340500y674200_x364600y677500 = krdzg.load_classified_krdz(
            ...     tile_xy=[(340500, 674200), (364600, 677500)])
            >>> krdz_x340500y674200_x364600y677500.shape
            (40, 7)
            >>> krdz_up = krdzg.load_classified_krdz(direction='up')
            >>> krdz_up.shape
            (9808, 7)
            >>> krdz_up_top_202004 = krdzg.load_classified_krdz(
            ...     pcd_date='202004', direction='up', element=['left top', 'right top'])
            >>> krdz_up_top_202004.shape
            (1961, 7)
            >>> krdz_up_top_202004.Element.unique().tolist()
            ['LeftTopOfRail', 'RightTopOfRail']
            >>> krdz_dat = krdzg.load_classified_krdz()
            >>> krdz_dat.shape
            (19629, 7)
        """

        tbl_name = f'"{self.KRDZ_SCHEMA_NAME}"."{self.KRDZ_CL_TABLE_NAME}"'
        query = f'SELECT * FROM {tbl_name}'

        query = add_sql_query_xy_condition(sql_query=query, tile_xy=tile_xy)

        query = add_sql_query_date_condition(sql_query=query, data_date=pcd_date)

        if direction is not None:
            direction_ = find_valid_names(direction, self.DIRECTIONS)[0]
            query += f' {"AND" if "WHERE" in query else "WHERE"} "Direction"=\'{direction_}\''

        if element is not None:
            element_ = find_valid_names(element, self.ELEMENTS)
            elem_cond = f'=\'{element_[0]}\'' if len(element_) == 1 else f' IN {tuple(element_)}'
            query += f' {"AND" if "WHERE" in query else "WHERE"} "Element"' + elem_cond

        if self.db_instance is None:
            self.db_instance = TrackFixityDB()

        clfd_rail_head_data = self.db_instance.read_sql_query(sql_query=query, method='tempfile')

        clfd_rail_head_data['geometry'] = clfd_rail_head_data['geometry'].map(shapely.wkt.loads)

        if as_dict:
            classified_rail_head_dict = {}

            key_cols = ['Year', 'Month', 'Tile_X', 'Tile_Y']
            for key, value in clfd_rail_head_data.groupby(key_cols):

                value.drop(columns=key_cols, inplace=True)
                # import collections
                dat_dict = {}  # collections.defaultdict(list)

                for i in value.index:
                    k = value.Direction[i] + '_' + value.Element[i]
                    dat_dict[k] = value.geometry[i]

                classified_rail_head_dict[key] = dat_dict

            clfd_rail_head_data = classified_rail_head_dict.copy()

        return clfd_rail_head_data
