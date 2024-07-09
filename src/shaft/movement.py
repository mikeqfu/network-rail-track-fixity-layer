"""
This module calculates/measures track movement in both vertical and horizontal plane, thus
determining and quantifying the fixity along both the vertical and horizontal axes of a track.
"""

import functools
import gc
import itertools
import multiprocessing
import os
import re
import shutil
import tempfile
import time
import webbrowser

import folium
import folium.plugins
import matplotlib.pyplot as plt
import natsort
import pandas as pd
import seaborn as sns
import shapely.ops
import shapely.wkt
from pyhelpers._cache import _print_failure_msg
from pyhelpers.dirs import cd
from pyhelpers.geom import osgb36_to_wgs84
from pyhelpers.ops import confirmed, is_url_connectable, split_list_by_size, swap_cols
from pyhelpers.store import save_figure
from pyhelpers.text import get_acronym
from shapely.geometry import GeometryCollection, LineString, MultiPoint

from src.shaft.krdz_gear import KRDZGear
from src.shaft.sec_utils import *
from src.utils.dgn import fix_folium_float_image
from src.utils.general import TrackFixityDB, abs_max, abs_min, cd_docs_source, find_valid_names
from src.utils.geometry import drop_z, make_a_polyline, point_projected_to_line


class TrackMovement(KRDZGear):
    """
    Calculate track movement - the target to be predicted.

    With the preprocessed point cloud data and relying on the classes
    :class:`~src.shaft.pcd_handler.PCDHandler` and :class:`~src.shaft.krdz_gear.KRDZGear`,
    this class leverages the KRDZ data to calculate track movement
    in both horizontal and vertical directions.
    """

    #: Descriptive name of the class.
    NAME: str = 'Calculator for the track movement'

    #: Name of the schema for storing the calculated track movement data.
    SCHEMA_NAME: str = 'TrackMovement'

    def __init__(self, elr='ECM8', db_instance=None):
        """
        :param elr: Engineer's Line Reference; defaults to ``'ECM8'``.
        :type elr: str
        :param db_instance: PostgreSQL database instance; defaults to ``None``.
        :type db_instance: TrackFixityDB | None

        :ivar str elr: Engineer's Line Reference.

        **Examples**::

            >>> from src.shaft import TrackMovement
            >>> tm = TrackMovement()
            >>> tm.NAME
            'Calculation of track movement'

        .. figure:: ../_images/tm_schema.*
            :name: tm_schema
            :align: center
            :width: 100%

            Snapshot of the *TrackMovement* schema.
        """

        super().__init__(db_instance=db_instance)

        self.elr = elr

    @staticmethod
    def split_section(section, unit_length=1, coarsely=False, use_original_points=True,
                      to_geoms=False):
        """
        Split a track section into a number of subsections,
        with each having approximately equal lengths.

        :param section: Track section (represented as a polyline).
        :type section: numpy.ndarry | LineString
        :param unit_length: Length of each subsection; defaults to ``1`` (in metre).
        :type unit_length: int | float
        :param coarsely: Whether to split the section in a coarse way; defaults to ``False``.
        :type coarsely: bool
        :param use_original_points: Whether to use original points as splitters;
            defaults to ``False``.
        :type use_original_points: bool
        :param to_geoms: Whether to transform the obtained geometry object (``GeometryCollection``)
            to its iterable form (``GeometrySequence``); defaults to ``True``.
        :type to_geoms: bool
        :return: A sequence of approximately equal-length subsections of the polyline object.
        :rtype: list | GeometryCollection

        **Examples**::

            >>> from src.shaft import TrackMovement
            >>> from src.shaft.sec_utils import weld_subsections
            >>> tm = TrackMovement()

        **Example 1** - A single tile::

            >>> tile_xy = (340500, 674200)
            >>> # Top of the left rail in the up direction within tile (340500, 674200) in 10/2019
            >>> krdz_ult_x340500y674200_201910 = tm.load_classified_krdz(
            ...     tile_xy=tile_xy, pcd_date='201910', direction='up', element='left top')
            >>> ult_201910 = krdz_ult_x340500y674200_201910.loc[0, 'geometry']
            >>> type(ult_201910)
            shapely.geometry.linestring.LineString
            >>> print("~%.4fm" % ult_201910.length)
            ~105.9960m
            >>> # Get subsections, with each being approximately 10-metre long
            >>> ult_201910_subs = tm.split_section(section=ult_201910, unit_length=10)
            >>> type(ult_201910_subs)
            shapely.geometry.collection.GeometryCollection
            >>> len(ult_201910_subs.geoms)
            11
            >>> for sub in ult_201910_subs.geoms:
            ...     print(sub.wkt[:50], "... ~%.4fm" % sub.length)
            LINESTRING Z (340599.367 674259.012 32.211, 340598 ... ~9.9989m
            LINESTRING Z (340589.994 674255.53 32.203, 340589. ... ~8.9999m
            LINESTRING Z (340581.557 674252.397 32.2, 340580.6 ... ~9.9995m
            LINESTRING Z (340572.183 674248.916 32.203, 340571 ... ~9.9990m
            LINESTRING Z (340562.811 674245.431 32.208, 340561 ... ~9.0000m
            LINESTRING Z (340554.375 674242.295 32.221, 340553 ... ~9.9992m
            LINESTRING Z (340545.002 674238.812 32.246, 340544 ... ~9.0000m
            LINESTRING Z (340536.566 674235.676 32.271, 340535 ... ~10.0005m
            LINESTRING Z (340527.192 674232.192 32.296, 340526 ... ~9.9990m
            LINESTRING Z (340517.82 674228.7070000001 32.323,  ... ~9.0000m
            LINESTRING Z (340509.384 674225.571 32.348, 340508 ... ~9.9999m

        **Example 2** - Multiple tiles::

            >>> tile_xy_list = [(360000, 677100), (360100, 677100), (360200, 677100)]
            >>> # Right top of rail in the down direction within the above tiles in April 2020
            >>> krdz_drt_multitiles_202004 = tm.load_classified_krdz(
            ...     tile_xy=tile_xy_list, pcd_date='202004', direction='down', element='right top')
            >>> krdz_drt_multitiles_202004
               Year  Month  Tile_X  Tile_Y Direction         Element                      geometry
            0  2020      4  360200  677100      Down  RightTopOfRail  LINESTRING Z (360200.16 6...
            1  2020      4  360000  677100      Down  RightTopOfRail  LINESTRING Z (360000.349 ...
            2  2020      4  360100  677100      Down  RightTopOfRail  LINESTRING Z (360100.773 ...
            >>> # Sum of the lengths of the three subsections
            >>> krdz_drt_multitiles_202004['geometry'].map(lambda ls: ls.length).sum()
            232.95245581071197
            >>> # Weld the three subsections into a single section
            >>> drt_202004 = weld_subsections(krdz_drt_multitiles_202004, start=(360000, 677100))
            >>> # Total length (where the additional ~2 metres are from the two "joints")
            >>> drt_202004.length
            234.9511338346051
            >>> # Divide the welded section into subsections of ~10 metres each
            >>> drt_202004_subs = tm.split_section(section=drt_202004, unit_length=10)
            >>> len(drt_202004_subs.geoms)
            24
            >>> for sub in drt_202004_subs.geoms:
            ...     print(sub.wkt[:50], "... ~%.4fm" % sub.length)
            LINESTRING Z (360000.349 677122.4300000001 28.067, ... ~10.0003m
            LINESTRING Z (360009.735 677125.8810000001 27.993, ... ~10.0008m
            LINESTRING Z (360019.116 677129.347 27.935, 360020 ... ~8.9996m
            LINESTRING Z (360027.556 677132.471 27.88, 360028. ... ~9.9980m
                ...                                            ...
            LINESTRING Z (360175.425 677185.098 27.27, 360176. ... ~9.9974m
            LINESTRING Z (360184.929 677188.2 27.258, 360185.8 ... ~9.9965m
            LINESTRING Z (360194.445 677191.262 27.251, 360195 ... ~8.9960m
            LINESTRING Z (360203.018 677193.988 27.248, 360203 ... ~9.9978m
            LINESTRING Z (360212.557 677196.982 27.244, 360213 ... ~9.9976m

        **Illustration**::

            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gs
            from matplotlib.offsetbox import AnchoredText
            from pyhelpers.settings import mpl_preferences

            mpl_preferences(backend='TkAgg', font_name='Times New Roman')

            fig = plt.figure(figsize=(7, 8), constrained_layout=True)
            mgs = gs.GridSpec(3, 1, figure=fig)

            colours = plt.get_cmap('tab10').colors

            # Original separate subsections
            ax1 = fig.add_subplot(mgs[0, :], aspect='equal', adjustable='box')
            for i in krdz_drt_multitiles_202004.index:
                ls = krdz_drt_multitiles_202004.geometry[i]
                ls_xs, ls_ys = ls.coords.xy
                tile_xy = tuple(krdz_drt_multitiles_202004.loc[i, ['Tile_X', 'Tile_Y']].values)
                ax1.plot(ls_xs, ls_ys, linewidth=1, color=colours[i], label=f'{tile_xy}')
            ax1.set_xlabel('Easting', fontsize=13, labelpad=5)
            ax1.set_ylabel('Northing', fontsize=13, labelpad=5)
            ax1.set_title('(a) Original subsections', y=0, pad=-55)
            ax1.legend()

            # Welded section
            ax2 = fig.add_subplot(mgs[1, :], aspect='equal', adjustable='box')
            welded_xs, welded_ys = drt_202004.coords.xy
            ax2.plot(welded_xs, welded_ys, linewidth=1, color=colours[i + 1])
            ax2.set_xlabel('Easting', fontsize=13, labelpad=5)
            ax2.set_ylabel('Northing', fontsize=13, labelpad=5)
            ax2.set_title('(b) Welded section', y=0, pad=-55)

            # Unit subsections
            ax3 = fig.add_subplot(mgs[2, :], aspect='equal', adjustable='box')
            for unit_sub in drt_202004_subs.geoms:
                sub_xs, sub_ys = unit_sub.coords.xy
                ax3.plot(sub_xs, sub_ys, linewidth=1)
            anchored_text = AnchoredText('Unit length = ~10m', loc=2)
            ax3.add_artist(anchored_text)
            ax3.set_xlabel('Easting', fontsize=13, labelpad=5)
            ax3.set_ylabel('Northing', fontsize=13, labelpad=5)
            ax3.set_title('(c) Unit subsections', y=0, pad=-55)

            # from pyhelpers.store import save_figure
            # fig_pathname = "docs/source/_images/tm_split_section_demo"
            # save_figure(fig, f"{fig_pathname}.svg", verbose=True)
            # save_figure(fig, f"{fig_pathname}.pdf", verbose=True)

        .. figure:: ../_images/tm_split_section_demo.*
            :name: tm_split_section_demo
            :align: center
            :width: 80%

            Welded section and unit subsections of the top of the right rail of a track section
            in the down direction within the tiles
            (360000, 677100), (360100, 677100) and (360200, 677100).
        """

        subsections = split_section(
            section=section, unit_length=unit_length, coarsely=coarsely,
            use_original_points=use_original_points, to_geoms=to_geoms)

        return subsections

    @staticmethod
    def illustrate_unit_displacement(unit_sec_orig, unit_sec_shifted, len_offset=False,
                                     element_label=None, add_title=False, **kwargs):
        """
        Illustrate the lateral and vertical displacements of a (small) track section.

        :param unit_sec_orig: Track section (i.e. original position of a track at an earlier time).
        :type unit_sec_orig: LineString
        :param unit_sec_shifted: Track section (i.e. position the track at a later time).
        :type unit_sec_shifted: LineString
        :param len_offset: Whether to len_offset the data of longer length by the shorter one;
            defaults to ``False``.
        :type len_offset: bool
        :param element_label: Label of the rail head's element to be illustrated;
            defaults to ``None``.
        :type element_label: str | None
        :param add_title: Whether to add a title to the plot; defaults to ``False``.
        :type add_title: bool
        :param kwargs: [Optional] additional parameters for the function
            :func:`~src.shaft.sec_utils.length_offset`.
        :type kwargs: float | int | None

        **Examples**::

            >>> from src.shaft.movement import TrackMovement
            >>> from src.shaft.sec_utils import length_offset, rearrange_line_points
            >>> from shapely.geometry import LineString
            >>> from pyhelpers.settings import mpl_preferences
            >>> mpl_preferences(backend='TkAgg')
            >>> tm = TrackMovement()
            >>> # Left top of rail in the up direction within tile (340500, 674200)
            >>> tile_xy = (340500, 674200)
            >>> direction = 'Up'
            >>> element = 'Left Top'
            >>> krdz_ult_x340500y674200 = tm.load_classified_krdz(tile_xy, None, direction, element)
            >>> ult_201910_ = krdz_ult_x340500y674200.query('Year == 2019')['geometry'].iloc[0]
            >>> type(ult_201910_)
            shapely.geometry.linestring.LineString
            >>> ult_201910_.length
            105.9959864289052
            >>> ult_202004_ = krdz_ult_x340500y674200.query('Year == 2020')['geometry'].iloc[0]
            >>> type(ult_202004_)
            shapely.geometry.linestring.LineString
            >>> ult_202004_.length
            105.99856695815154
            >>> # Offset the section length and rearrange the lines
            >>> ult_201910, ult_202004 = length_offset(ult_201910_, ult_202004_)
            >>> ult_201910, ult_202004 = rearrange_line_points(ult_201910, ult_202004)
            >>> # In this example, the two sections remain as is due to little difference in lengths
            >>> ult_201910.length
            105.9959864289052
            >>> ult_202004.length
            105.99856695815154
            >>> # Divide into ~one-metre subsections
            >>> ult_201910_subs, ult_202004_subs = map(tm.split_section, [ult_201910, ult_202004])
            >>> # An example 1m subsection of the top of left rail in the up direction in Oct. 2019
            >>> ult_201910_1m = ult_201910_subs.geoms[100]
            >>> # A corresponding 1m subsection in Apr. 2020
            >>> ult_202004_1m = ult_202004_subs.geoms[100]
            >>> annot = 'top of the left rail'
            >>> tm.illustrate_unit_displacement(ult_201910_1m, ult_202004_1m, element_label=annot)
            >>> # from pyhelpers.store import save_fig
            >>> # fig_pathname = "docs/source/_images/tm_illust_unit_disp_x340500y674200_ult"
            >>> # save_fig(f"{fig_pathname}.svg", transparent=True, verbose=True)
            >>> # save_fig(f"{fig_pathname}.pdf", transparent=True, verbose=True)

        .. figure:: ../_images/tm_illust_unit_disp_x340500y674200_ult.*
            :name: tm_illust_unit_disp_x340500y674200_ult
            :align: center
            :width: 100%

            Movement of the top of the left rail of a ~one-metre track section
            in the up direction within the Tile (340500, 674200).
        """

        if len_offset:
            sect_original_, sect_shifted_ = length_offset(
                section_original=unit_sec_orig, section_shifted=unit_sec_shifted, **kwargs)
        else:
            sect_original_, sect_shifted_ = unit_sec_orig, unit_sec_shifted

        sect_original_, sect_shifted_ = rearrange_line_points(sect_original_, sect_shifted_)

        fig = plt.figure(figsize=(8, 7))  # constrained_layout=True
        ax = fig.add_subplot(projection='3d')
        ax.ticklabel_format(useOffset=False)

        colours = plt.colormaps.get_cmap(cmap='tab10').colors

        # Transform the input data to array
        original_xyz, xyz_shifted = map(
            lambda x: np.array(x.coords), [sect_original_, sect_shifted_])

        # Plot the two track sections
        label = 'Rail head' if element_label is None else element_label.capitalize()
        ax.plot3D(
            original_xyz[:, 0], original_xyz[:, 1], original_xyz[:, 2],
            linewidth=2, color=colours[0], label=f'{label} of a one-metre section (Oct. 2019)')
        ax.plot3D(
            xyz_shifted[:, 0], xyz_shifted[:, 1], xyz_shifted[:, 2],
            linewidth=2, color=colours[1], label=f'{label} of the one-metre section (Apr. 2020)')

        ax.tick_params(axis='x', which='major', pad=4)
        ax.tick_params(axis='y', which='major', pad=8)
        ax.tick_params(axis='z', which='major', pad=12)

        # Find the section centroid from the earlier data
        original_c_xyz = sect_original_.interpolate(0.5, normalized=True)

        # In the shifted data, find the nearest point to the original centroid
        _, c_xy_shifted = shapely.ops.nearest_points(g1=original_c_xyz, g2=sect_shifted_)

        # The intersection point of 'lateral shift' and 'vertical shift'
        mapping_xyz = Point(np.append(np.array(c_xy_shifted.coords), original_c_xyz.z))

        # Plot the lateral shift
        lateral_shift_line = np.array(LineString([original_c_xyz, mapping_xyz]).coords)
        ax.plot3D(
            lateral_shift_line[:, 0], lateral_shift_line[:, 1], lateral_shift_line[:, 2],
            color=colours[2], linewidth=1, linestyle='--', label='Lateral displacement')

        # Find the intersection point of a perpendicular line
        # _, c_xyz_shifted = point_projected_to_line(point=original_c_xyz, line=sect_shifted_)
        c_xyz_shifted_ = min(
            shapely.ops.split(sect_shifted_, MultiPoint(sect_shifted_.coords)).geoms,
            key=lambda x: x.centroid.distance(original_c_xyz))
        _, c_xyz_shifted = point_projected_to_line(point=original_c_xyz, line=c_xyz_shifted_)

        # Plot the vertical shift
        vertical_shift_line = np.array(LineString([mapping_xyz, c_xyz_shifted]).coords)
        ax.plot3D(
            vertical_shift_line[:, 0], vertical_shift_line[:, 1], vertical_shift_line[:, 2],
            color=colours[3], linewidth=1, linestyle='--', label='Vertical displacement')

        # Plot the perpendicular line
        perp_line = np.array(LineString([original_c_xyz, c_xyz_shifted]).coords)
        ax.plot3D(
            perp_line[:, 0], perp_line[:, 1], perp_line[:, 2],
            color=colours[4], linewidth=1, linestyle='--', label='Perpendicular line')

        ax.scatter3D(
            xs=original_c_xyz.x, ys=original_c_xyz.y, zs=original_c_xyz.z,
            marker='o', s=35, color=colours[0], label='Centroid of the track section (Oct. 2019)')
        ax.scatter3D(
            xs=c_xyz_shifted.x, ys=c_xyz_shifted.y, zs=c_xyz_shifted.z,
            marker='o', s=35, color=colours[1],
            label='Intersection point (Apr. 2020) of the perpendicular line\n'
                  'through the centroid (Oct. 2019)')

        ax.set_xlabel('Easting', fontsize=13, labelpad=12)
        ax.set_ylabel('Northing', fontsize=13, labelpad=22)
        ax.set_zlabel('Elevation', fontsize=13, labelpad=22)

        if add_title:
            plt.title("An example of the lateral and vertical displacements being calculated.")

        ax.legend(loc='best', numpoints=1, ncol=1)  # fontsize=11, , fancybox=True, framealpha=0.6

        # plt.subplots_adjust(left=0.09, bottom=0.05, right=1, top=1)
        fig.tight_layout()

    @staticmethod
    def calculate_unit_subsection_displacement(unit_sec_orig, unit_sec_shifted, ref_obj):
        # noinspection PyShadowingNames
        """
        Calculate the displacement of a (short) section (of approx. one metre)
        between two different times.

        :param unit_sec_orig: A unit track section's position at an earlier time.
        :type unit_sec_orig: numpy.ndarray | LineString
        :param unit_sec_shifted: (Almost) the same unit track section' position at a later time.
        :type unit_sec_shifted: numpy.ndarray | LineString
        :param ref_obj: A reference object for identifying direction of the displacement.
        :type ref_obj: numpy.ndarray | LineString
        :return: Both lateral and vertical displacements.
        :rtype: tuple

        **Examples**::

            >>> from src.shaft import TrackMovement
            >>> from src.shaft.sec_utils import rearrange_line_points
            >>> from src.utils import geom_distance
            >>> from pyhelpers.settings import mpl_preferences
            >>> mpl_preferences(backend='TkAgg')
            >>> tm = TrackMovement()
            >>> # KRDZ data within the Tile (357500, 677700)
            >>> krdz_x357500y677700 = tm.load_classified_krdz(tile_xy=(357500, 677700))
            >>> krdz_x357500y677700.head()
               Year  ...                                           geometry
            0  2019  ...  LINESTRING Z (357599.131 677701.85 28.137, 357...
            1  2019  ...  LINESTRING Z (357599.144 677701.882 28.118, 35...
            2  2019  ...  LINESTRING Z (357599.686 677703.249 27.988, 35...
            3  2019  ...  LINESTRING Z (357599.672 677703.214 27.976, 35...
            4  2019  ...  LINESTRING Z (357599.409 677702.55 28.063, 357...
            [5 rows x 7 columns]
            >>> # Top of the left rail in the up direction within (357500, 677700)
            >>> expr = 'Direction=="Up" and Element=="LeftTopOfRail"'
            >>> ult_x357500y677700 = krdz_x357500y677700.query(expr).set_index(['Year', 'Month'])
            >>> # Data of Oct. 2019
            >>> ult201910 = ult_x357500y677700.loc[(2019, 10), 'geometry']
            >>> # Data of Apr. 2020
            >>> ult202004 = ult_x357500y677700.loc[(2020, 4), 'geometry']
            >>> # Adjust the order of points in the linestring
            >>> ult201910, ult202004 = rearrange_line_points(ult201910, ult202004)
            >>> # Divide the linestring into one-metre subsections
            >>> ult201910_subs = tm.split_section(section=ult201910, use_original_points=False)
            >>> len(ult201910_subs.geoms)
            109
            >>> ult202004_subs = tm.split_section(section=ult202004, use_original_points=False)
            >>> len(ult202004_subs.geoms)
            109
            >>> # An example of a one-metre subsection of the top of left rail in the up direction
            >>> ult201910_1m = ult201910_subs.geoms[50]
            >>> ult201910_1m.length
            0.9914271527767892
            >>> print(ult201910_1m.wkt)
            LINESTRING Z (357553.2933530097 677720.7146399085 27.88829200114355, 357552.3846181...
            >>> # The one-metre subsection in Apr. 2020 corresponding to the example in Oct. 2019
            >>> ult202004_1m = min(ult202004_subs.geoms, key=geom_distance(ult201910_1m.centroid))
            >>> ult202004_1m.length
            0.9914087119426246
            >>> print(ult202004_1m.wkt)
            LINESTRING Z (357553.3663216831 677720.6891952073 27.88775019639971, 357552.4577717...
            >>> # Illustrate the distance between the two one-metre data
            >>> annot = 'top of left rail'
            >>> tm.illustrate_unit_displacement(ult201910_1m, ult202004_1m, element_label=annot)
            >>> # from pyhelpers.store import save_fig
            >>> # fig_pathname = "docs/source/_images/tm_calc_unit_subsect_disp_x357500y677700_demo"
            >>> # save_fig(f"{fig_pathname}.svg", transparent=True, verbose=True)
            >>> # save_fig(f"{fig_pathname}.pdf", transparent=True, verbose=True)

        .. figure:: ../_images/tm_calc_unit_subsect_disp_x357500y677700_demo.*
            :name: tm_calc_unit_subsect_disp_x357500y677700_demo
            :align: center
            :width: 90%

            Movement of the top of the left rail of a ~one-metre track section
            in the up direction within the Tile (357500, 677700).

        .. code-block:: python

            >>> # Get a reference object (for identifying the direction of displacement)
            >>> ref_expr = 'Year==2019 and Direction=="Up" and Element=="Centre"'
            >>> ref_objects = krdz_x357500y677700.query(ref_expr)['geometry'].iloc[0]
            >>> # Lateral and vertical displacements
            >>> displacement_data = tm.calculate_unit_subsection_displacement(
            ...     ult201910_1m, ult202004_1m, ref_objects)
            >>> lateral_disp, vertical_disp, abs_min_disp = displacement_data
            >>> # Lateral displacements of one end, centroid and the other end of the 1-m section
            >>> lateral_disp
            [-0.006304874255657488, -0.0058847316089858095, -0.0060948028789788]
            >>> # Vertical displacements of one end, centroid and the other end of the 1-m section
            >>> vertical_disp
            [-0.0008873096940130941, -0.0009624371630553282, -0.0009248734283608409]
        """

        displacement_data = calculate_unit_subsection_displacement(
            unit_sect_orig=unit_sec_orig, unit_sect_shifted=unit_sec_shifted, ref_obj=ref_obj)

        return displacement_data

    def _get_subsec_geoms(self, section_original, section_shifted, unit_length=1, len_offset=True,
                          multi_processes=False, **kwargs):
        if len_offset:
            temp_original, temp_shifted = length_offset(
                section_original=section_original, section_shifted=section_shifted, buf_type=1)
            section_data = rearrange_line_points(temp_original, temp_shifted)
        else:
            section_data = rearrange_line_points(section_original, section_shifted)

        kwargs.update(dict(unit_length=unit_length, to_geoms=True))
        if not multi_processes:
            original_geoms, shifted_geoms = map(
                functools.partial(self.split_section, **kwargs), section_data)
        else:
            with multiprocessing.Pool(processes=os.cpu_count() - 1) as p:
                original_geoms, shifted_geoms = p.map(
                    functools.partial(split_section, **kwargs), section_data)

        if len(original_geoms) != len(shifted_geoms):
            cen_original = np.vstack([np.array(x.centroid.coords) for x in original_geoms])
            cen_shifted = np.vstack([np.array(x.centroid.coords) for x in shifted_geoms])

            base_geoms = max([original_geoms, shifted_geoms], key=len)
            assert isinstance(base_geoms, shapely.geometry.base.GeometrySequence)

            if len(original_geoms) < len(shifted_geoms):
                shifted_geoms = find_closest_subsections(cen_original, cen_shifted, base_geoms)
            else:
                original_geoms = find_closest_subsections(cen_shifted, cen_original, base_geoms)

        return original_geoms, shifted_geoms

    @staticmethod
    def _calc_disp_stats(movement_data, unit_length, subsect_len, disp_column_names, rolling):
        stats_calc = ['mean', 'std', abs_min, abs_max]

        sub_len = subsect_len // unit_length

        if rolling:
            movement_data_ = movement_data[disp_column_names].rolling(window=sub_len)

            movement_data_ = movement_data_.aggregate(stats_calc).dropna()
            movement_data_.index = range(len(movement_data_))

            idx = range(len(movement_data) - sub_len + 1)

        else:
            movement_data_ = movement_data[disp_column_names].groupby(
                np.arange(len(movement_data)) // sub_len).aggregate(stats_calc)

            idx = range(0, len(movement_data), sub_len)

        stats_column_names = ['_'.join(col) for col in movement_data_.columns]
        movement_data_.columns = stats_column_names

        subsection_coords = [
            shapely.ops.linemerge(movement_data['subsection'].iloc[i:i + sub_len]).coords
            for i in idx]
        movement_data_.insert(loc=0, column='subsection', value=subsection_coords)
        movement_data_['subsection'] = movement_data_['subsection'].map(LineString)

        return movement_data_

    def calculate_section_movement(self, section_original, section_shifted, ref_obj, unit_length=1,
                                   subsect_len=10, len_offset=True, rolling=False,
                                   multi_processes=False, **kwargs):
        """
        Calculate average displacement about the movement of a track section
        for every subsection of a given length.

        :param section_original: A track section's original position of a track at an earlier time.
        :type section_original: numpy.ndarray | LineString
        :param section_shifted: (Almost) the same track section's position of at a later time.
        :type section_shifted: numpy.ndarray | LineString
        :param ref_obj: A reference object for identifying direction of the displacement.
        :type ref_obj: numpy.ndarray | LineString
        :param len_offset: Whether to balance the section lengths;
            mostly, offset the data of longer length by the shorter one; defaults to ``True``.
        :type len_offset: bool
        :param unit_length: Length (in metre) of each subsection; defaults to ``1``.
        :type unit_length: int | float
        :param subsect_len: Length (in metre) of a subsection for which movement is calculated;
            defaults to ``10``.
        :type subsect_len: int
        :param rolling: Whether to calculate the statistics on a rolling basis; 
            defaults to ``False``.
        :type rolling: bool
        :param multi_processes: Whether to use multiple CPUs (see also `multiprocessing.Pool()`_);
            defaults to ``False``.
        :type multi_processes: bool
        :param kwargs: [Optional] parameters of the method 
            :meth:`~src.shaft.movement.TrackMovement.split_section`.
        :type kwargs: bool
        :return: Basic statistics of track movement.
        :rtype: pandas.DataFrame

        .. _`multiprocessing.Pool()`:
            https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool

        **Examples**::

            >>> from src.shaft import TrackMovement
            >>> from src.shaft.sec_utils import rearrange_line_points, length_offset
            >>> from src.utils import geom_distance
            >>> from pyhelpers.settings import mpl_preferences
            >>> import functools
            >>> mpl_preferences(backend='TkAgg')
            >>> tm = TrackMovement()
            >>> # KRDZ data of the rail heads in the down direction within the Tile (357500, 677700)
            >>> krdz_x357500y677700_d = tm.load_classified_krdz((357500, 677700), direction='down')
            >>> krdz_x357500y677700_d.head()
               Year  ...                                           geometry
            0  2019  ...  LINESTRING Z (357599.306 677705.436 27.895, 35...
            1  2019  ...  LINESTRING Z (357599.319 677705.469 27.875, 35...
            2  2019  ...  LINESTRING Z (357599.867 677706.834 27.743, 35...
            3  2019  ...  LINESTRING Z (357599.853 677706.8 27.731, 3575...
            4  2019  ...  LINESTRING Z (357599.587 677706.136 27.819, 35...
            [5 rows x 7 columns]
            >>> # Top of the right rail in the down direction within (357500, 677700)
            >>> expr = 'Element=="RightTopOfRail"'
            >>> drt_x357500y677700 = krdz_x357500y677700_d.query(expr).set_index(['Year', 'Month'])
            >>> # Data of Oct. 2019
            >>> drt201910_ = drt_x357500y677700.loc[(2019, 10), 'geometry']
            >>> # Data of Apr. 2020
            >>> drt202004_ = drt_x357500y677700.loc[(2020, 4), 'geometry']
            >>> # Adjust the order of points in the linestring
            >>> drt201910, drt202004 = rearrange_line_points(drt201910_, drt202004_)
            >>> drt201910.length
            108.06188326010438
            >>> drt202004.length
            109.06372793933721
            >>> # Get a reference object (for identifying the direction of displacement)
            >>> ref_expr = 'Year==2019 and Element=="Centre"'
            >>> ref_objects = krdz_x357500y677700_d.query(ref_expr).geometry.values[0]
            >>> print(ref_objects.wkt)
            LINESTRING Z (357599.587 677706.1360000001 27.819, 357598.656 677706.501 27.814, 35...
            >>> # Illustrate an example 1-metre subsection
            >>> drt201910_subs, drt202004_subs = map(
            ...     functools.partial(tm.split_section, use_original_points=False),
            ...     length_offset(drt201910, drt202004))
            >>> drt201910_1m = drt201910_subs.geoms[15]
            >>> drt202004_1m = min(drt202004_subs.geoms, key=geom_distance(drt201910_1m.centroid))
            >>> annot = 'top of the right rail'
            >>> tm.illustrate_unit_displacement(drt201910_1m, drt202004_1m, element_label=annot)
            >>> # from pyhelpers.store import save_fig
            >>> # fig_pathname = "docs/source/_images/tm_calc_sect_movement_x357500y677700_drt"
            >>> # save_fig(f"{fig_pathname}.svg", transparent=True, verbose=True)
            >>> # save_fig(f"{fig_pathname}.pdf", transparent=True, verbose=True)

        .. figure:: ../_images/tm_calc_sect_movement_x357500y677700_drt.*
            :name: tm_calc_sect_movement_x357500y677700_drt
            :align: center
            :width: 100%

            An exmaple of the lateral and vertical displacements of the top of right rail
            of a ~one-metre subsection in the down direction within the Tile (357500, 677700).

        .. code-block:: python

            >>> # Average displacements for every 10-metre subsection
            >>> drt_movement_x357500y677700 = tm.calculate_section_movement(
            ...     section_original=drt201910, section_shifted=drt202004, ref_obj=ref_objects,
            ...     subsect_len=10)
            >>> drt_movement_x357500y677700.head()
               pseudo_id  ... vertical_displacement_cen_abs_max
            0          0  ...                         -0.001012
            1          1  ...                         -0.000910
            2          2  ...                         -0.001012
            3          3  ...                         -0.001305
            4          4  ...                         -0.001405
            [5 rows x 38 columns]
            >>> drt_movement_x357500y677700.shape
            (11, 38)
            >>> # Rolling average of every 10 metres
            >>> drt_movement_x357500y677700_ra = tm.calculate_section_movement(
            ...     section_original=drt201910, section_shifted=drt202004, ref_obj=ref_objects,
            ...     subsect_len=10, rolling=True)
            >>> drt_movement_x357500y677700_ra.shape
            (99, 38)
        """

        kwargs.update(
            dict(unit_length=unit_length, len_offset=len_offset, multi_processes=multi_processes))
        original_geoms, shifted_geoms = self._get_subsec_geoms(
            section_original, section_shifted, **kwargs)

        if not multi_processes:
            movement_dat = [
                self.calculate_unit_subsection_displacement(
                    unit_sec_orig=sec_orig, unit_sec_shifted=sec_shifted, ref_obj=ref_obj)
                for sec_orig, sec_shifted in zip(original_geoms, shifted_geoms)]

        else:
            with multiprocessing.Pool(processes=os.cpu_count() - 1) as p:
                movement_dat = p.starmap(
                    functools.partial(calculate_unit_subsection_displacement, ref_obj=ref_obj),
                    zip(original_geoms, shifted_geoms))
            gc.collect()

        disp_col_names = [x + '_displacement' for x in ['lateral', 'vertical', 'abs_minimum']]
        movement_dat_ = pd.DataFrame(movement_dat, columns=disp_col_names)  # Create a dataframe

        disp_dat = [
            pd.DataFrame(
                movement_dat_[col].tolist(), columns=[col + x for x in ('_a', '_b', '_cen')])
            for col in disp_col_names[:2]]
        disp_data = pd.concat(disp_dat, axis=1)

        for col in disp_col_names[:2]:
            movement_dat_[col] = movement_dat_[col].map(np.mean)

        disp_column_names = disp_col_names + disp_data.columns.tolist()

        movement_data = pd.concat([movement_dat_, disp_data], axis=1)
        movement_data.insert(
            loc=0, column='subsection', value=pd.Series([x.coords for x in original_geoms]))
        movement_data['subsection'] = movement_data['subsection'].map(LineString)

        if subsect_len > 1:
            movement_data = self._calc_disp_stats(
                movement_data=movement_data, unit_length=unit_length, subsect_len=subsect_len,
                disp_column_names=disp_column_names, rolling=rolling)

        movement_data = movement_data.reset_index().rename(columns={'index': 'pseudo_id'})

        return movement_data

    def weld_classified_krdz(self, element, direction, pcd_date, tile_xy=None):
        # noinspection PyShadowingNames
        """
        Weld the classified KRDZ rail head data for all tiles.

        :param element: Element of rail head.
        :type element: str
        :param direction: Railway direction; options include ``{'up', 'down'}``.
        :type direction: str
        :param pcd_date: Date of the point cloud data.
        :type pcd_date: str | int
        :param tile_xy: Easting and northing of a tile for the point cloud data;
            defaults to ``None``.
        :type tile_xy: tuple | list | str | None
        :return: Polyline geometry of the target track.
        :rtype: LineString

        **Examples**::

            >>> from src.shaft import TrackMovement
            >>> tm = TrackMovement()
            >>> element = 'Left Top'
            >>> direction = 'Up'
            >>> welded_ult_201910 = tm.weld_classified_krdz(element, direction, pcd_date='201910')
            >>> welded_ult_201910.length
            76299.56835581265
            >>> welded_ult_202004 = tm.weld_classified_krdz(element, direction, pcd_date='202004')
            >>> welded_ult_202004.length
            76300.69543101029

        **Illustration**::

            import matplotlib.pyplot as plt
            from pyhelpers.settings import mpl_preferences

            mpl_preferences(backend='TkAgg', font_name='Times New Roman')

            fig = plt.figure(figsize=(11, 5), constrained_layout=True)
            ax = fig.add_subplot(aspect='equal', adjustable='box')
            ax.grid()

            xs_o, ys_o = welded_ult_201910.coords.xy
            ax.plot(xs_o, ys_o, lw=1, marker='o', markersize=5, label='LeftTopOfRail 201910')
            xs_s, ys_s = welded_ult_202004.coords.xy
            ax.plot(xs_s, ys_s, lw=1, marker='^', markersize=5, label='LeftTopOfRail 202004')

            ax.legend()

            # from pyhelpers.store import save_figure
            # fig_pathname_ = "docs/source/_images/tm_weld_classified_krdz_ult_demo"
            # save_figure(fig, f"{fig_pathname_}.svg", transparent=True, verbose=True)
            # save_figure(fig, f"{fig_pathname_}.pdf", transparent=True, verbose=True)

        .. figure:: ../_images/tm_weld_classified_krdz_ult_demo.*
            :name: tm_weld_classified_krdz_ult_demo
            :align: center
            :width: 100%

            The top of left rail in the up direction (10/2019 vs. 04/2020).
        """

        krdz_dat = self.load_classified_krdz(
            tile_xy=tile_xy, pcd_date=pcd_date, direction=direction, element=element)

        if len(krdz_dat) == 1:
            welded_section = krdz_dat.geometry.values[0]

        else:
            tiles_xys = krdz_dat[['Tile_X', 'Tile_Y']].to_numpy()
            tiles_convex_hull = self.get_tiles_convex_hull(pcd_tiles=tiles_xys, as_array=True)
            start_tile_x, start_tile_y = tiles_convex_hull[
                -1 if re.search(r'up', direction, re.IGNORECASE) else 1]

            start_line = krdz_dat.query(f'Tile_X == {start_tile_x} and Tile_Y == {start_tile_y}')

            centroids_xys = krdz_dat.geometry.map(
                lambda x: np.array(shapely.ops.transform(drop_z, x).centroid.coords[0]))
            krdz_dat[['centroid_x', 'centroid_y']] = pd.DataFrame(centroids_xys.map(list).to_list())

            start_centroid = krdz_dat.geometry[start_line.index].values[0].centroid

            cen_xys = np.array(centroids_xys.to_list())
            cen_xys_sorted = make_a_polyline(
                points_sequence=cen_xys, start_point=start_centroid, as_geom=False)

            krdz_dat_ = krdz_dat.set_index(['centroid_x', 'centroid_y']).loc[cen_xys_sorted]
            krdz_dat_.index = range(len(krdz_dat_))

            # Sort the KRDZ points of the obtained line `krdz_dat_`
            welded_section = sort_line_points(line_geoms=krdz_dat_.geometry, as_geom=True)

        return welded_section

    def calculate_movement(self, element, direction, tile_xy=None, pcd_dates=None, ref_obj=None,
                           len_offset=True, unit_length=1, subsect_len=10, rolling=False,
                           **kwargs):
        """
        Calculate average displacement about the movement of a given track section.

        :param element: Element of rail head, such as left or right top of rail or running edge.
        :type element: str
        :param direction: Railway direction, such as up or down direction.
        :type direction: str
        :param tile_xy: Easting (X) and northing (Y) of the geographic Cartesian coordinates
            for a tile; defaults to ``None``.
        :type tile_xy: tuple | list | str
        :param pcd_dates: Dates of the point cloud data to compare; defaults to ``None``.
        :type pcd_dates: list | tuple | None
        :param ref_obj: Reference object for identifying direction of the displacement.
        :type ref_obj: numpy.ndarray | LineString
        :param len_offset: Whether to balance the section lengths;
            mostly, offset the data of longer length by the shorter one; defaults to ``True``.
        :type len_offset: bool
        :param unit_length: Length (in metre) of each subsection; defaults to ``1``.
        :type unit_length: int | float
        :param subsect_len: Length (in metre) of a subsection for which movement is calculated;
            defaults to ``10``.
        :type subsect_len: int
        :param rolling: Whether to calculate the statistics on a rolling basis;
            defaults to ``False``.
        :type rolling: bool
        :param kwargs: [Optional] parameters of the method
            :meth:`~src.shaft.movement.TrackMovement.calculate_section_movement`.
        :type kwargs: bool
        :return: Average displacement about the movement of a given track section.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft import TrackMovement
            >>> tm = TrackMovement()
            >>> ult_movements = tm.calculate_movement(
            ...     element='Left Top', direction='Up', multi_processes=True)
            >>> ult_movements
                  pseudo_id  ... vertical_displacement_cen_abs_max
            0             0  ...                         -0.003203
            1             1  ...                         -0.002798
            2             2  ...                         -0.002200
            3             3  ...                         -0.001305
            4             4  ...                         -0.001000
            ...         ...  ...                               ...
            7624       7624  ...                         -0.001239
            7625       7625  ...                         -0.000825
            7626       7626  ...                         -0.000912
            7627       7627  ...                          0.000766
            7628       7628  ...                          0.001267
            [7629 rows x 38 columns]
            >>> ult_movements_co = tm.calculate_movement(
            ...     element='Left Top', direction='Up', coarsely=True, multi_processes=True)
            >>> ult_movements_co
                  pseudo_id  ... vertical_displacement_cen_abs_max
            0             0  ...                         -0.003203
            1             1  ...                         -0.002798
            2             2  ...                         -0.002200
            3             3  ...                         -0.001305
            4             4  ...                         -0.001000
            ...         ...  ...                               ...
            7626       7626  ...                         -0.001239
            7627       7627  ...                         -0.000912
            7628       7628  ...                         -0.000735
            7629       7629  ...                          0.001267
            7630       7630  ...                          0.001178
            [7631 rows x 38 columns]
        """

        pcd_dates_ = self.check_pcd_dates(pcd_dates, len_req=2)

        date_original, date_shifted = natsort.natsorted(pcd_dates_)

        section_original = self.weld_classified_krdz(
            element=element, direction=direction, pcd_date=date_original, tile_xy=tile_xy)

        section_shifted = self.weld_classified_krdz(
            element=element, direction=direction, pcd_date=date_shifted, tile_xy=tile_xy)

        # Get a reference object - Centre line of the track section
        if ref_obj is None:
            ref_object = self.weld_classified_krdz(
                element='Centre', direction=direction, pcd_date=date_original, tile_xy=tile_xy)
        else:
            ref_object = ref_obj

        movements_data = self.calculate_section_movement(
            section_original=section_original, section_shifted=section_shifted,
            ref_obj=ref_object, len_offset=len_offset, unit_length=unit_length,
            subsect_len=subsect_len, rolling=rolling, **kwargs)

        return movements_data

    def get_valid_table_names(self, element=None, direction=None, pcd_dates=None, ret_input=False):
        """
        Get valid name(s) of the table(s) for storing track movement data of unit sections.

        :param element: Element of rail head, e.g. left/right top of rail or running edge;
            defaults to ``None``.
        :type element: str | list | None
        :param direction: Railway direction, e.g. up and down directions; defaults to ``None``.
        :type direction: str | list | None
        :param pcd_dates: Dates of the point cloud data to compare; defaults to ``None``.
        :type pcd_dates: list | tuple | None
        :param ret_input: Whether to return validated names of the input
            ``element`` and ``direction``; defaults to ``False``.
        :return: Valid name(s) of the table(s) (for storing track movement data of unit sections)
            and, when ``ret_input=True``, validated names of ``element``, ``direction`` and
            ``pcd_dates``.
        :rtype: list | tuple

        **Examples**::

            >>> from src.shaft import TrackMovement
            >>> tm = TrackMovement()
            >>> valid_table_names = tm.get_valid_table_names()
            >>> valid_table_names
            ['Up_LeftTopOfRail_201910_202004',
             'Up_LeftRunningEdge_201910_202004',
             'Up_RightTopOfRail_201910_202004',
             'Up_RightRunningEdge_201910_202004',
             'Up_Centre_201910_202004',
             'Down_LeftTopOfRail_201910_202004',
             'Down_LeftRunningEdge_201910_202004',
             'Down_RightTopOfRail_201910_202004',
             'Down_RightRunningEdge_201910_202004',
             'Down_Centre_201910_202004']
            >>> valid_table_names = tm.get_valid_table_names(element=['left top', 'right top'])
            >>> valid_table_names
            ['Up_LeftTopOfRail_201910_202004',
             'Up_RightTopOfRail_201910_202004',
             'Down_LeftTopOfRail_201910_202004',
             'Down_RightTopOfRail_201910_202004']
        """

        pcd_dates_ = self.check_pcd_dates(pcd_dates, len_req=2)

        elements = find_valid_names(element, self.ELEMENTS)
        directions = find_valid_names(direction, self.DIRECTIONS)

        tbls = ['_'.join(list(x) + pcd_dates_) for x in itertools.product(*[directions, elements])]

        if ret_input:
            tbls = tbls, elements, directions, pcd_dates_

        return tbls

    def import_unit_movement(self, element=None, direction=None, pcd_dates=None,
                             confirmation_required=True, verbose=True, rolling=False, **kwargs):
        """
        Import the data of track movement of unit sections into the project database.

        :param element: Element of rail head, e.g. left/right top of rail or running edge;
            defaults to ``None``.
        :type element: str | None
        :param direction: Railway direction, e.g. up and down directions; defaults to ``None``.
        :type direction: str | None
        :param pcd_dates: Dates of the point cloud data to compare; defaults to ``None``.
        :type pcd_dates: list | tuple | None
        :param confirmation_required: Whether asking for confirmation to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in console; defaults to ``True``.
        :type verbose: bool | int
        :param rolling: Whether to calculate the statistics on a rolling basis;
            defaults to ``False``.
        :type rolling: bool
        :param kwargs: [Optional] parameters of `pyhelpers.dbms.PostgreSQL.import_data`_.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/
            _generated/pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.shaft import TrackMovement
            >>> tm = TrackMovement()
            >>> tm.import_unit_movement(if_exists='replace')
            To import unit movement data into the following tables in the schema "TrackMovement":
                "Up_LeftTopOfRail_201910_202004"
                "Up_LeftRunningEdge_201910_202004"
                "Up_RightTopOfRail_201910_202004"
                "Up_RightRunningEdge_201910_202004"
                "Up_Centre_201910_202004"
                "Down_LeftTopOfRail_201910_202004"
                "Down_LeftRunningEdge_201910_202004"
                "Down_RightTopOfRail_201910_202004"
                "Down_RightRunningEdge_201910_202004"
                "Down_Centre_201910_202004"
            ? [No]|Yes: yes
            Importing the data ...
                Up_LeftTopOfRail_201910_202004 ... Done.
                Up_LeftRunningEdge_201910_202004 ... Done.
                Up_RightTopOfRail_201910_202004 ... Done.
                Up_RightRunningEdge_201910_202004 ... Done.
                Up_Centre_201910_202004 ... Done.
                Down_LeftTopOfRail_201910_202004 ... Done.
                Down_LeftRunningEdge_201910_202004 ... Done.
                Down_RightTopOfRail_201910_202004 ... Done.
                Down_RightRunningEdge_201910_202004 ... Done.
                Down_Centre_201910_202004 ... Done.

        .. figure:: ../_images/tm_ult_201910_202004_tbl.*
            :name: tm_ult_201910_202004_tbl
            :align: center
            :width: 100%

            Snapshot of the "TrackMovement"."Up_LeftTopOfRail_201910_202004" table.
        """

        elements = find_valid_names(element, self.ELEMENTS)
        directions = find_valid_names(direction, self.DIRECTIONS)
        directions_elements = list(itertools.product(*[directions, elements]))

        pcd_dates_ = self.check_pcd_dates(pcd_dates, len_req=2)

        tbl_names = ['_'.join(list(x) + pcd_dates_) for x in directions_elements]
        if rolling:
            tbl_names = [x + '_RA' for x in tbl_names]

        if len(tbl_names) == 1:
            print_msg = f"table \"{self.SCHEMA_NAME}\".\"{tbl_names[0]}\""
        else:
            print_msg = "following tables in the schema \"{}\":\n\t\"{}\"".format(
                self.SCHEMA_NAME, f'"\n\t"'.join(tbl_names))

        confirm_msg = f"To import unit movement data into the {print_msg}\n?"
        if confirmed(confirm_msg, confirmation_required=confirmation_required):
            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                if confirmation_required:
                    print("Importing the data ... ")
                else:
                    print(confirm_msg.replace("\n?", "---------------") + "\nIn progress ... ")

            for (direct, elem), table_name in zip(directions_elements, tbl_names):
                try:
                    if verbose:
                        print(f"\t{table_name}", end=" ... ")

                    movements_data = self.calculate_movement(
                        element=elem, direction=direct, pcd_dates=pcd_dates, unit_length=1,
                        subsect_len=1, rolling=rolling)

                    self.db_instance.import_data(
                        data=movements_data, schema_name=self.SCHEMA_NAME, table_name=table_name,
                        method=self.db_instance.psql_insert_copy, confirmation_required=False,
                        **kwargs)

                    if verbose:
                        print("Done.")

                except Exception as e:
                    _print_failure_msg(e)

        else:
            if verbose:
                print("Cancelled.")

    def load_movement(self, element=None, direction=None, pcd_dates=None, subsect_len=10,
                      rolling=False, keep_pseudo_id=False, **kwargs):
        """
        Load (and calculate) average displacement about the movement of a given track section.

        :param element: Element of rail head, e.g. left/right top of rail or running edge;
            defaults to ``None``.
        :type element: str | list | None
        :param direction: Railway direction, e.g. up and down directions; defaults to ``None``.
        :type direction: str | list | None
        :param pcd_dates: Dates of the point cloud data to compare; defaults to ``None``.
        :type pcd_dates: list | tuple | None
        :param subsect_len: Length (in metre) of a subsection for which movement is calculated;
            defaults to ``10``.
        :type subsect_len: int
        :param rolling: Whether to calculate the statistics on a rolling basis;
            defaults to ``False``.
        :type rolling: bool
        :param keep_pseudo_id: Whether to keep ``pseudo_id`` for unit subsections;
            defaults to ``False``.
        :type keep_pseudo_id: bool
        :param kwargs: [Optional] parameters of `pyhelpers.dbms.PostgreSQL.read_sql_query`_.
        :return: Average displacement about the movement of a given track section.
        :rtype: dict

        .. _`pyhelpers.dbms.PostgreSQL.read_sql_query`:
            https://pyhelpers.readthedocs.io/en/latest/
            _generated/pyhelpers.dbms.PostgreSQL.read_sql_query.html

        **Examples**::

            >>> from src.shaft import TrackMovement
            >>> tm = TrackMovement()
            >>> ult_movement = tm.load_movement(element='Left Top', direction='Up')
            >>> type(ult_movement)
            dict
            >>> list(ult_movement.keys())
            ['Up_LeftTopOfRail_201910_202004']
            >>> ult_movement['Up_LeftTopOfRail_201910_202004']
                                                  subsection  ...  vertical_displacement_b_abs_max
            0     LINESTRING Z (399428.96 653473.9 34.442...  ...                        -0.003203
            1     LINESTRING Z (399434.5201166851 653482....  ...                        -0.002794
            2     LINESTRING Z (399440.0608295194 653490....  ...                        -0.003202
            3     LINESTRING Z (399445.5700898784 653498....  ...                        -0.001803
            4     LINESTRING Z (399451.0456262118 653507....  ...                        -0.001201
                                                         ...  ...                              ...
            7625  LINESTRING Z (340180.451306418 674103.1...  ...                        -0.001000
            7626  LINESTRING Z (340171.0809418422 674099....  ...                        -0.001650
            7627  LINESTRING Z (340161.7096778383 674096....  ...                        -0.000914
            7628  LINESTRING Z (340152.3387371677 674092....  ...                        -0.000737
            7629  LINESTRING Z (340142.9690238989 674089....  ...                         0.001267
            [7630 rows x 37 columns]
            >>> ult_movement_ra = tm.load_movement(element='Left Top', direction='Up', rolling=True)
            >>> ult_movement_ra['Up_LeftTopOfRail_201910_202004_RA']
                                                  subsection  ...  vertical_displacement_b_abs_max
            0      LINESTRING Z (399428.96 653473.9 34.44...  ...                        -0.003203
            1      LINESTRING Z (399429.5183848582 653474...  ...                        -0.003203
            2      LINESTRING Z (399430.0744669316 653475...  ...                        -0.003203
            3      LINESTRING Z (399430.6317003522 653476...  ...                        -0.003001
            4      LINESTRING Z (399431.1884705836 653477...  ...                        -0.003001
                                                         ...  ...                              ...
            76286  LINESTRING Z (340146.7167791607 674090...  ...                         0.001265
            76287  LINESTRING Z (340145.7800034651 674090...  ...                         0.001265
            76288  LINESTRING Z (340144.8429013437 674089...  ...                         0.001265
            76289  LINESTRING Z (340143.9061259775 674089...  ...                         0.001265
            76290  LINESTRING Z (340142.9690238989 674089...  ...                         0.001267
            [76291 rows x 37 columns]
            >>> trk_movement_100sl = tm.load_movement(subsect_len=100, keep_pseudo_id=True)
            >>> len(trk_movement_100sl)
            10
            >>> list(trk_movement_100sl.keys())
            ['Up_LeftTopOfRail_201910_202004',
             'Up_LeftRunningEdge_201910_202004',
             'Up_RightTopOfRail_201910_202004',
             'Up_RightRunningEdge_201910_202004',
             'Up_Centre_201910_202004',
             'Down_LeftTopOfRail_201910_202004',
             'Down_LeftRunningEdge_201910_202004',
             'Down_RightTopOfRail_201910_202004',
             'Down_RightRunningEdge_201910_202004',
             'Down_Centre_201910_202004']
            >>> trk_movement_100sl['Up_LeftTopOfRail_201910_202004']
                                                    pseudo_id  ... vertical_displacement_b_abs_max
            0    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12...  ...                       -0.003203
            1    [100, 101, 102, 103, 104, 105, 106, 107, ...  ...                       -0.005816
            2    [200, 201, 202, 203, 204, 205, 206, 207, ...  ...                       -0.003543
            3    [300, 301, 302, 303, 304, 305, 306, 307, ...  ...                       -0.002782
            4    [400, 401, 402, 403, 404, 405, 406, 407, ...  ...                       -0.004486
            ..                                            ...  ...                             ...
            758  [75800, 75801, 75802, 75803, 75804, 75805...  ...                       -0.006993
            759  [75900, 75901, 75902, 75903, 75904, 75905...  ...                       -0.004793
            760  [76000, 76001, 76002, 76003, 76004, 76005...  ...                       -0.004802
            761  [76100, 76101, 76102, 76103, 76104, 76105...  ...                       -0.004801
            762  [76200, 76201, 76202, 76203, 76204, 76205...  ...                       -0.001740
            [763 rows x 38 columns]
        """

        table_names = self.get_valid_table_names(
            element=element, direction=direction, pcd_dates=pcd_dates)

        tbl_names = [f'"{self.SCHEMA_NAME}"."{x}"' for x in table_names]

        if self.db_instance is None:
            self.db_instance = TrackFixityDB()

        movement_data_list = []

        for tbl_name in tbl_names:
            query = f'SELECT * FROM {tbl_name} ORDER BY "pseudo_id"'
            unit_movement_data = self.db_instance.read_sql_query(sql_query=query, **kwargs)

            unit_movement_data.subsection = unit_movement_data.subsection.map(shapely.wkt.loads)

            if subsect_len > 1:
                disp_column_names = [
                    col for col in unit_movement_data.columns if 'displacement' in col]
                stats_calc = ['mean', 'std', abs_min, abs_max]

                disp_dat = unit_movement_data[disp_column_names]

                if rolling:
                    movement_data = disp_dat.rolling(window=subsect_len).agg(stats_calc).dropna()

                    # indices = range(len(unit_movement_data) - subsect_len + 1)
                    indices = [
                        range(i, j + 1)
                        for i, j in zip(range(len(movement_data)), movement_data.index)]

                    movement_data.index = range(len(movement_data))

                else:
                    group_range = np.arange(len(disp_dat)) // subsect_len

                    movement_data = disp_dat.groupby(group_range).aggregate(stats_calc)

                    # indices = range(0, len(unit_movement_data), subsect_len)
                    indices = list(
                        split_list_by_size(unit_movement_data.pseudo_id.to_list(), subsect_len))

                gc.collect()

                movement_data.columns = ['_'.join(col) for col in movement_data.columns]

                # subsection_geom = [
                #     shapely.ops.linemerge(unit_movement_data.subsection.iloc[i:i + subsect_len])
                #     for i in indices]
                subsection_geom = [
                    shapely.ops.linemerge(unit_movement_data.subsection.iloc[idx]).coords
                    for idx in indices]
                movement_data.insert(loc=0, column='subsection', value=subsection_geom)
                movement_data['subsection'] = movement_data['subsection'].map(LineString)

                if keep_pseudo_id:
                    movement_data.insert(loc=0, column='pseudo_id', value=indices)

                gc.collect()

                movement_data_list.append(movement_data)

            else:
                movement_data_list.append(unit_movement_data)

        if rolling:
            table_names = [x + '_RA' for x in table_names]

        movement_data_dict = dict(zip(table_names, movement_data_list))

        return movement_data_dict

    @staticmethod
    def view_movement_violin_plot(data, fig_size=(10, 6), save_as=None, dpi=600, verbose=False,
                                  **kwargs):
        # noinspection PyShadowingNames
        """
        Create a violin plot of the track movement.

        :param data: Data of the track movement.
        :type data: pandas.DataFrame
        :param fig_size: Figure size; defaults to ``(10, 6)``.
        :type fig_size: tuple[float, float]
        :param save_as: File format that the figure is saved as; defaults to ``None``.
        :type save_as: str | list | None
        :param dpi: DPI for saving image; defaults to ``600``.
        :type dpi: int | None
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters of the function
            `pyhelpers.store.save_figure`_.

        .. _`pyhelpers.store.save_figure`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/pyhelpers.store.save_figure.html

        **Examples**::

            >>> from src.shaft import TrackMovement
            >>> from pyhelpers.settings import mpl_preferences
            >>> mpl_preferences(backend='TkAgg')
            >>> tm = TrackMovement()
            >>> movement_data_dict = tm.load_movement(
            ...     element='Left Top', direction='Up', subsect_len=10)
            >>> movement_data = movement_data_dict['Up_LeftTopOfRail_201910_202004']
            >>> column_names = ['lateral_displacement_mean', 'vertical_displacement_mean']
            >>> data = movement_data[column_names] * 1000
            >>> tm.view_movement_violin_plot(data, fig_size=(10, 6))
            >>> # tm.view_movement_violin_plot(data, save_as=".svg", verbose=True)

        .. figure:: ../_images/tm_view_movement_violin_plot.*
            :name: tm_view_movement_violin_plot
            :align: center
            :width: 80%

            Violin plot of the average track movement for every 10-m section.
        """

        column_names = ['displacement', 'displacement_direction']

        data['lateral_displacement_direction'] = data['lateral_displacement_mean'].map(
            lambda x: 'Rightwards' if x < 0 else 'Leftwards')
        dat1 = data[['lateral_displacement_mean', 'lateral_displacement_direction']]
        dat1.columns = column_names

        data['vertical_displacement_direction'] = data['vertical_displacement_mean'].map(
            lambda x: 'Downwards' if x < 0 else 'Upwards')
        dat2 = data[['vertical_displacement_mean', 'vertical_displacement_direction']]
        dat2.columns = column_names

        dat = pd.concat([dat1, dat2], axis=0)

        # data[column_names] = data[column_names].abs().values

        # 10-m average lateral displacement (mm)
        fig = plt.figure(figsize=fig_size)  # , constrained_layout=True

        ax = fig.add_subplot()  # aspect='equal', adjustable='box'
        ax.xaxis.grid(linestyle='-', linewidth=1, alpha=.3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_position(('outward', 8))
        ax.spines['bottom'].set_position(('outward', 5))

        sns.violinplot(
            data=dat, x='displacement', y='displacement_direction', orient='h',
            hue='displacement_direction', legend=False,
            palette=['#FFFFBF', '#ABDDA4', '#D4E2D4', '#FFD460'], ax=ax)
        # sns.violinplot(
        #     data=data, x='lateral_displacement_mean', y='lateral_shift_direction', orient='h',
        #     inner='quartile', palette=['#FFFFBF', '#ABDDA4'], ax=ax1)
        # ax1.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('10-m average displacement (mm)', fontsize=17, labelpad=8)
        ax.xaxis.set_ticks(ax.get_xticks())
        ax.xaxis.set_ticklabels(
            ['%.1f' % x if x < 0 else ('+%.1f' % x if x > 0 else 0.0) for x in ax.get_xticks()])
        ax.set_ylabel('Displacement direction', fontsize=17, labelpad=9)

        ax.tick_params(axis='both', which='major', labelsize=16)

        ax.set_xlim(xmin=-20, xmax=20)

        fig.tight_layout()
        fig.subplots_adjust(left=0.17, bottom=0.12, right=0.96, top=0.96)

        if save_as:
            for ext in {save_as, ".svg", ".pdf"}:
                path_to_fig = cd_docs_source("_images", "tm_view_movement_violin_plot" + ext)
                kwargs.update({'transparent': True, 'dpi': dpi, 'verbose': verbose})
                save_figure(fig, path_to_fig, **kwargs)

    def _get_map_view_location(self):
        """
        **Examples**::

            >>> from src.shaft import TrackMovement
            >>> tm = TrackMovement()
            >>> tm._get_map_view_location()
            [55.91240547442317, -2.479967560695688]
        """

        if self.db_instance is None:
            self.db_instance = TrackFixityDB()

        pcd_tiles = self.db_instance.read_sql_query(
            sql_query=f'SELECT * FROM "{KRDZGear.SCHEMA_NAME}"."{KRDZGear.TILES_TABLE_NAME}"',
            method='tempfile')

        pcd_tiles_lonlat = pcd_tiles.drop_duplicates(subset=['LAZ_Filename'])['Tile_LonLat']

        location_polygon = self.get_tiles_convex_hull(pcd_tiles_lonlat)

        location_ = np.array(location_polygon.centroid.coords[0])
        # location = location_.tolist()
        # location.reverse()
        location = location_.tolist()[::-1]

        return location

    @staticmethod
    def _get_heatmap_view_data(geometry):
        centroids = geometry.map(
            lambda x: shapely.ops.transform(drop_z, x.interpolate(0.5, normalized=True)).coords[0])

        centroids_xy = np.array(centroids.to_list())
        centroids_lonlat = np.array(osgb36_to_wgs84(centroids_xy[:, 0], centroids_xy[:, 1])).T

        return centroids_lonlat

    @staticmethod
    def _add_north_arrow(m):
        img_url = "https://creazilla-store.fra1.digitaloceanspaces.com/cliparts/3867221/" \
                  "north-arrow-clipart-md.png"

        if is_url_connectable(img_url):
            north_arrow = img_url
        else:
            north_arrow = 'demos\\images\\north_arrow.png'

        folium.plugins.FloatImage(
            image=north_arrow, bottom=8, left=1,  # bottom=1, left=8
            width='22px').add_to(m)

    def _open_html_in_browser(self, m, movement_data_key, pcd_dates, update):
        hm_filename = "heatmap_view_{}_{}.html".format(
            get_acronym(movement_data_key, capitals_in_words=True).lower(), "_".join(pcd_dates))

        path_to_hm = cd(self.DATA_DIR, self.elr, hm_filename)

        if not os.path.exists(path_to_hm) or update:
            m.save(path_to_hm)

            fix_folium_float_image(path_to_hm)

            shutil.copyfile(path_to_hm, cd_docs_source("_static", hm_filename))
            webbrowser.open(path_to_hm)

            # from PIL import Image
            # import io
            # # noinspection PyProtectedMember
            # img_data = m._to_png(delay=5)
            # img = Image.open(io.BytesIO(img_data))
            # img.save(cd_docs_source("_images", hm_filename.replace(".html", ".png")))

        else:
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp_name = tmp.name + '.html'
            try:
                m.save(tmp_name)

                fix_folium_float_image(tmp_name)

                webbrowser.open(tmp_name)
                time.sleep(2)

            finally:
                tmp.close()
                os.unlink(tmp_name)

    def view_heatmap(self, element, direction, subsect_len=100, pcd_dates=None, col_names=None,
                     scale=10 ** 3, open_html=True, update=False, **kwargs):
        """
        Create a heat map view of the track movement.

        :param element: Element of rail head, e.g. left/right top of rail or running edge;
            defaults to ``None``.
        :type element: str | list | None
        :param direction: Railway direction, e.g. up and down directions; defaults to ``None``.
        :type direction: str | list | None
        :param subsect_len: Length (in metre) of a subsection for which movement is calculated;
            defaults to ``10``.
        :type subsect_len: int
        :param pcd_dates: Dates of the point cloud data to compare; defaults to ``None``.
        :type pcd_dates: list | tuple | None
        :param col_names: Names of columns to be viewed.
        :type col_names: list | None
        :param scale: Scale of the calculated values of displacements on the heat map;
            defaults to ``10 ** 3``.
        :type scale: int | float
        :param open_html: Whether to open a local HTML file in an explorer to view the map;
            defaults to ``True``.
        :type open_html: bool
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param kwargs: [Optional] parameters of the method
            :meth:`~src.shaft.movement.TrackMovement.load_movement`.

        **Examples**::

            >>> from src.shaft import TrackMovement
            >>> tm = TrackMovement()
            >>> # tm.view_heatmap(element='Left Top', direction='Up', subsect_len=10, update=True)
            >>> tm.view_heatmap(element='Left Top', direction='Up', subsect_len=10)

        .. raw:: html

            <iframe title="Heatmap for movement of the top of the left rail in the up direction."
                src="../_static/heatmap_view_ultor_201910_202004.html"
                marginwidth="0" marginheight="0" scrolling="no"
                style="width:100%; height:600px; border:0; overflow:hidden;">
            </iframe>

        .. only:: html

            .. figure:: ../_images/blank.png
                :name: blank_heatmap_view_ultor_201910_202004
                :align: center
                :width: 0%

                Heatmap for movement of the top of the left rail in the up direction
                (on a 10-metre basis).

        .. only:: latex

            .. figure:: ../_images/heatmap_view_ultor_201910_202004.*
                :name: heatmap_view_ultor_201910_202004
                :align: center
                :width: 100%

                Heatmap for movement of the top of the left rail in the up direction
                (on a 10-metre basis).
        """

        pcd_dates_ = self.check_pcd_dates(pcd_dates, len_req=2)

        movement_data_dict = self.load_movement(
            element=element, direction=direction, pcd_dates=pcd_dates_, subsect_len=subsect_len,
            **kwargs)
        key, movement_data = list(movement_data_dict.items())[0]
        key_ = ' - '.join(key.split('_')[:2])

        if col_names is None:
            dat_col_names = ['lateral_displacement_mean', 'vertical_displacement_mean']
        else:
            dat_col_names = col_names.copy()

        centroids_lonlat = self._get_heatmap_view_data(movement_data['subsection'])
        map_view_loc = self._get_map_view_location()

        # Optional param: tiles='CartoDB positron'
        m = folium.Map(location=map_view_loc, tiles=None, zoom_start=10, control_scale=True)
        folium.TileLayer('OpenStreetMap', name=f'{key_} (04/2020 vs. 10/2019)').add_to(m)

        folium.plugins.MiniMap(zoom_level_offset=-6).add_to(m)

        hm_common_args = dict(
            min_opacity=0.3, radius=15, blur=10, max_zoom=1, use_local_extrema=False)

        hm_lateral = folium.FeatureGroup(name='Lateral displacement').add_to(m)
        lateral_val = movement_data[dat_col_names[0]].abs().values * scale
        folium.plugins.HeatMap(
            data=list(zip(centroids_lonlat[:, 1], centroids_lonlat[:, 0], lateral_val)),
            name='Lateral displacement', **hm_common_args
        ).add_to(hm_lateral)

        hm_vertical = folium.FeatureGroup(name='Vertical displacement', show=False).add_to(m)
        vertical_val = movement_data[dat_col_names[1]].abs().values * scale
        folium.plugins.HeatMap(
            data=list(zip(centroids_lonlat[:, 1], centroids_lonlat[:, 0], vertical_val)),
            name='Vertical displacement', **hm_common_args
        ).add_to(hm_vertical)

        v_popup = folium.FeatureGroup(name='Average displacement (metres)', control=False)
        v_means = movement_data[dat_col_names].to_numpy()

        popup_loc = swap_cols(centroids_lonlat.copy(), 1, 0, as_list=True)

        for i in range(len(v_means)):
            a, b = map(
                lambda p: '(%.2f, %.2f)' % (p.x, p.y), movement_data.subsection[i].boundary.geoms)
            sect_len = np.round(movement_data.subsection[i].length)
            v = np.round(v_means[i], 4)
            folium.CircleMarker(
                location=popup_loc[i],
                popup=folium.Popup(
                    f'Length: {sect_len} m<br>'
                    f'Start: {a}<br>'
                    f'End: {b}<br>'
                    f'Lateral shift: {v[0]} m<br>'
                    f'Vertical shift: {v[1]} m',
                    max_width=500),
                radius=15,
                opacity=0,
                fill=True,
                fill_opacity=0).add_to(v_popup)
        m.add_child(v_popup)

        folium.LayerControl(collapsed=False).add_to(m)

        self._add_north_arrow(m)

        if open_html:
            self._open_html_in_browser(
                m, movement_data_key=key_, pcd_dates=pcd_dates_, update=update)

        return m
