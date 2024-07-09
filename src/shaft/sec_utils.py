"""
This following utilities are used specifically for facilitating the calculation of track movement.
"""

import copy

import numpy as np
import shapely.ops
import shapely.wkt
from scipy.spatial import cKDTree
from shapely.geometry import GeometryCollection, LineString, Point

from src.utils.errors import InvalidSubsectionLength
from src.utils.general import paired_next
from src.utils.geometry import drop_y, drop_z, make_a_polyline, point_projected_to_line


def _trim_offset(section_original, section_shifted, buf_type, buf_dist):
    if buf_type == 1:
        buf_original = section_original.interpolate(distance=0.5, normalized=True).buffer(
            distance=(section_original.length / 2 if buf_dist is None else buf_dist))
        buf_shifted = section_shifted.interpolate(distance=0.5, normalized=True).buffer(
            distance=(section_shifted.length / 2 if buf_dist is None else buf_dist))

    else:
        buf_dist_ = (1 if buf_type == 2 else 0.1) if buf_dist is None else buf_dist
        buf_original = section_original.buffer(distance=buf_dist_, cap_style=buf_type)
        buf_shifted = section_shifted.buffer(distance=buf_dist_, cap_style=buf_type)

    buffer = buf_original.intersection(buf_shifted)

    sect_original = section_original.intersection(buffer)
    sect_shifted = section_shifted.intersection(buffer)

    return sect_original, sect_shifted


def _offset_pt_line(pt_line, line):
    # pt_line, line = section_original, section_shifted
    _, pt_of_line = shapely.ops.nearest_points(pt_line.centroid, line)
    # buf = pt_of_line.buffer(distance=pt_of_line.distance(pt_line.centroid))
    # pt_line_ = line.intersection(buf)
    return pt_of_line


def length_offset(section_original, section_shifted, min_diff=0.25, buf_type=1, buf_dist=None):
    """
    Offset the section lengths measured at two different times.

    :param section_original: Track section,
        representing the original position at an earlier time.
    :type section_original: LineString
    :param section_shifted: Track section, representing the position at a later time.
    :type section_shifted: LineString
    :param min_diff: The minimum difference between the input data lengths,
        beyond which the function proceeds; defaults to ``0.25``.
    :type min_diff: int | float
    :param buf_type: Buffer type;
        options include ``1`` (circle), ``2`` (flat) and ``3`` (square)
        (see `shapely.geometry.CAP_STYLE`_ for more details); defaults to ``1``.
    :type buf_type: int
    :param buf_dist: Radius of the buffer (see `shapely.geometry.buffer`_ for more details);
        defaults to ``None``.
    :type buf_dist: int | float
    :return: Data of balanced section lengths.
    :rtype: tuple

    .. _`shapely.geometry.CAP_STYLE`:
        https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.CAP_STYLE
    .. _`shapely.geometry.buffer`:
        https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.buffer

    **Examples**::

        >>> from src.shaft.sec_utils import length_offset
        >>> from src.shaft import KRDZGear
        >>> krdzg = KRDZGear()
        >>> # Top of the left rail in the up direction within the Tile (358300, 677500)
        >>> krdz_x358300y677500_dlt = krdzg.load_classified_krdz(
        ...     tile_xy=(358300, 677500), direction='down', element='left top')
        >>> krdz_x358300y677500_dlt
           Year  Month  Tile_X  Tile_Y Direction        Element                       geometry
        0  2019     10  358300  677500      Down  LeftTopOfRail  LINESTRING Z (358399.8 677...
        1  2020      4  358300  677500      Down  LeftTopOfRail  LINESTRING Z (358300.402 6...
        >>> # Top of the left rail in the up direction within (358300, 677500) in 10/2019
        >>> dlt_201910_ = krdz_x358300y677500_dlt.query('Year == 2019').geometry.values[0]
        >>> dlt_201910_.length
        105.06377258078923
        >>> print(dlt_201910_.wkt)
        LINESTRING Z (358399.8 677539.6090000001 31.357, 358398.872 677539.983 31.358, 3583...
        >>> # Top of the left rail in the up direction within (358300, 677500) in 04/2020
        >>> dlt_202004_ = krdz_x358300y677500_dlt.query('Year == 2020').geometry.values[0]
        >>> dlt_202004_.length
        105.02011737767278
        >>> print(dlt_202004_.wkt)
        LINESTRING Z (358300.402 677574.988 31.372, 358301.357 677574.693 31.38, 358302.313...
        >>> # Offset the lengths
        >>> dlt_201910, dlt_202004 = length_offset(dlt_201910_, dlt_202004_, buf_type=2)
        >>> dlt_201910.length
        104.54662324839607
        >>> print(dlt_201910.wkt)
        LINESTRING Z (358399.3203395722 677539.8023114224 31.35751687546096, 358398.872 677...
        >>> dlt_202004.length
        104.54631064536706
        >>> print(dlt_202004.wkt)
        LINESTRING Z (358300.8547005132 677574.8481605744 31.37579225560777, 358301.357 677...

    **Illustration**::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyhelpers.settings import mpl_preferences

        mpl_preferences(backend='TkAgg')

        # Two different buffers for the polyline of original data
        buf_o1 = dlt_201910_.interpolate(0.5, normalized=True).buffer(dlt_201910_.length / 2)
        buf_o2 = dlt_201910_.buffer(distance=0.1, cap_style=2)

        # Two different buffers for the polyline of shifted data
        buf_s1 = dlt_202004_.interpolate(0.5, normalized=True).buffer(dlt_202004_.length / 2)
        buf_s2 = dlt_202004_.buffer(distance=0.1, cap_style=2)

        # Intersections between the paired buffers of both the polylines
        buf_poly1, buf_poly2 = buf_o1.intersection(buf_s1), buf_o2.intersection(buf_s2)

        # Get array data of the position data and the buffers
        sect_o, sect_s = map(np.array, (dlt_201910_.coords, dlt_202004_.coords))
        buf1, buf2 = map(np.array, (buf_poly1.exterior.coords, buf_poly2.exterior.coords))

        # Illustration the buffers in R^2 plane - vertical view
        fig = plt.figure(figsize=(8, 7), constrained_layout=True)
        ax = fig.add_subplot(aspect='equal', adjustable='box')
        ax.ticklabel_format(useOffset=False)

        cm = plt.get_cmap('tab10').colors

        ax.plot(sect_o[:, 0], sect_o[:, 1], lw=5, alpha=0.3, label='original', color=cm[0])
        ax.plot(sect_s[:, 0], sect_s[:, 1], lw=5, alpha=0.3, label='shifted', color=cm[1])
        ax.plot(buf1[:, 0], buf1[:, 1], label='cap_style = 1', color=cm[2], linestyle='--')
        ax.plot(buf2[:, 0], buf2[:, 1], label='cap_style = 2', color=cm[3], linestyle='--')

        o_ = np.array(dlt_201910.intersection(buf_poly2).coords)
        s_ = np.array(dlt_202004.intersection(buf_poly2).coords)
        ax.plot(o_[:, 0], o_[:, 1], color=cm[4], label='original (retained)')
        ax.plot(s_[:, 0], s_[:, 1], color=cm[5], label='shifted (retained)')

        ax.legend()

        ax.set_xlabel('Easting', fontsize=13, labelpad=5)
        ax.set_ylabel('Northing', fontsize=13, labelpad=5)

        # from pyhelpers.store import save_figure
        # fig_pathname = "docs/source/_images/tm_length_offset_demo"
        # save_figure(fig, f"{fig_pathname}.svg", verbose=True)
        # save_figure(fig, f"{fig_pathname}.pdf", verbose=True)
        #
        # save_figure(fig, f"{fig_pathname}_zoomed_in.svg", verbose=True)
        # save_figure(fig, f"{fig_pathname}_zoomed_in.pdf", verbose=True)

    .. figure:: ../_images/sec_utils_length_offset_demo.*
        :name: sec_utils_length_offset_demo
        :align: center
        :width: 80%

        Illustration of offset buffers for the top of the left rail
        in the up direction within the Tile (358300, 677500).

    .. figure:: ../_images/sec_utils_length_offset_demo_zoomed_in.*
        :name: sec_utils_length_offset_demo_zoomed_in
        :align: center
        :width: 80%

        (Zoomed-in) Illustration of offset buffers for the top of the left rail
        in the up direction within the Tile (358300, 677500).
    """

    len_original, len_shifted = section_original.length, section_shifted.length

    section_original_, section_shifted_ = copy.copy(section_original), copy.copy(section_shifted)

    if abs(len_original - len_shifted) <= min_diff:
        # i.e. there is little difference in the lengths of the two polylines
        min_dist_1 = min(section_original.project(a) for a in section_shifted.boundary.geoms)
        min_dist_2 = min(section_shifted.project(b) for b in section_original.boundary.geoms)
        if max(min_dist_1, min_dist_2) > 0.5:
            section_original_, section_shifted_ = _trim_offset(
                section_original, section_shifted, buf_type, buf_dist)
    else:
        if any(x == 0 for x in (len_original, len_shifted)):
            if len_original == 0:
                section_original_ = section_original.centroid
                # section_shifted_ = _offset_pt_line(section_original, section_shifted)
                _, section_shifted_ = shapely.ops.nearest_points(section_original_, section_shifted)

            if len_shifted == 0:
                section_shifted_ = section_shifted.centroid
                # section_original_ = _offset_pt_line(section_shifted, section_original)
                _, section_original_ = shapely.ops.nearest_points(section_shifted_, section_original)

        else:
            section_original_, section_shifted_ = _trim_offset(
                section_original, section_shifted, buf_type, buf_dist)

    return section_original_, section_shifted_


def find_closest_subsections(cen1, cen2, sect_base):
    """
    Find the closest subsections.

    :param cen1: Centroid of one track subsection.
    :type cen1: numpy.ndarray
    :param cen2: Centroid of another track subsection.
    :type cen2: numpy.ndarray
    :param sect_base: The reference object.
        (In this project, it is the relatively longer track subsection).
    :type sect_base: shapely.geometry.base.GeometrySequence
    :return: The closest subsection given the shortest distance between ``cen1`` and ``cen2``,
        taking ``sect_base`` as the reference object.
    :rtype: shapely.geometry.base.GeometrySequence
    """

    cen2_ckdtree = cKDTree(cen2)

    _, indices = cen2_ckdtree.query(cen1, k=1)  # distances and indices
    sect_shifted_ = GeometryCollection([sect_base[i] for i in indices]).geoms

    return sect_shifted_


def weld_subsections(krdz_dat, start, as_geom=True):
    """
    Weld a number of track subsections together into a single section.

    :param krdz_dat: KRDZ data loaded from the project database.
    :type krdz_dat: pandas.DataFrame
    :param start: Point at which the polyline (i.e. welded track section) starts.
    :type start: tuple | list | Point
    :param as_geom: Whether to return a geometry object;
        when ``as_geom=False``, the method returns an array of coordinates;
        defaults to ``True``.
    :type as_geom: bool
    :return: A track section consisting of several subsections.
    :rtype: LineString | numpy.ndarray

    **Examples**::

        >>> from src.shaft.sec_utils import weld_subsections
        >>> from src.shaft import KRDZGear
        >>> krdzg = KRDZGear()
        >>> tiles_xys = [(360100, 677100), (360200, 677100), (360000, 677100)]
        >>> # Top of the right rail in the down direction (within the above tiles) in April 2020
        >>> krdz_dat_202004_drt = krdzg.load_classified_krdz(
        ...     tile_xy=tiles_xys, pcd_date='202004', direction='down', element='right top')
        >>> krdz_dat_202004_drt
           Year  Month  Tile_X  Tile_Y Direction         Element                      geometry
        0  2020      4  360200  677100      Down  RightTopOfRail  LINESTRING Z (360200.16 6...
        1  2020      4  360000  677100      Down  RightTopOfRail  LINESTRING Z (360000.349 ...
        2  2020      4  360100  677100      Down  RightTopOfRail  LINESTRING Z (360100.773 ...
        >>> # Sum of the lengths of the three subsections
        >>> krdz_dat_202004_drt['geometry'].map(lambda x: x.length).sum()
        232.95245581071197
        >>> drt_202004_welded = weld_subsections(krdz_dat_202004_drt, start=(360000, 677100))
        >>> # Total length (where the additional ~2 metres are from the two "joints")
        >>> drt_202004_welded.length
        234.9511338346051

    **Illustration**::

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gs
        from pyhelpers.settings import mpl_preferences

        mpl_preferences(backend='TkAgg', font_name='Times New Roman')

        fig = plt.figure(figsize=(8, 7), constrained_layout=True)
        mgs = gs.GridSpec(2, 1, figure=fig)

        colours = plt.get_cmap('tab10').colors

        # Original separate subsections
        ax1 = fig.add_subplot(mgs[0, :], aspect='equal', adjustable='box')
        for i in krdz_dat_202004_drt.index:
            ls = krdz_dat_202004_drt['geometry'][i]
            ls_xs, ls_ys = ls.coords.xy
            tile_xy = tuple(krdz_dat_202004_drt.loc[i, ['Tile_X', 'Tile_Y']].astype(int).tolist())
            ax1.plot(ls_xs, ls_ys, linewidth=1, color=colours[i], label=f'{tile_xy}')
        ax1.set_xlabel('Easting', fontsize=13, labelpad=5)
        ax1.set_ylabel('Northing', fontsize=13, labelpad=5)
        ax1.set_title('(a) Original subsections', y=0, pad=-55)
        ax1.legend()

        # Welded section
        ax2 = fig.add_subplot(mgs[1, :], aspect='equal', adjustable='box')
        welded_xs, welded_ys = drt_202004_welded.coords.xy
        ax2.plot(welded_xs, welded_ys, linewidth=1, color=colours[i + 1])
        ax2.set_xlabel('Easting', fontsize=13, labelpad=5)
        ax2.set_ylabel('Northing', fontsize=13, labelpad=5)
        ax2.set_title('(b) Welded section', y=0, pad=-55)

        # from pyhelpers.store import save_figure
        # fig_pathname = "docs/source/_images/tm_weld_subsections_demo"
        # save_figure(fig, f"{fig_pathname}.svg", verbose=True)
        # save_figure(fig, f"{fig_pathname}.pdf", verbose=True)

    .. figure:: ../_images/sec_utils_weld_subsections_demo.*
        :name: sec_utils_weld_subsections_demo
        :align: center
        :width: 80%

        Welded section of the top of the right rail of in the down direction within the tiles
        (360000, 677100), (360100, 677100) and (360200, 677100).
    """

    idx_col = ['Tile_X', 'Tile_Y']

    tiles_list = krdz_dat[idx_col].values.tolist()

    # Sort the subsections
    tiles_list_sorted = make_a_polyline(points_sequence=tiles_list, start_point=start, as_geom=False)

    # Get geometry objects in the sorted order
    polylines = krdz_dat.set_index(idx_col).loc[tiles_list_sorted]['geometry'].map(
        lambda x: np.array(x.coords))

    # Weld the subsections into a single section
    welded_section = np.unique(np.vstack(polylines.values), axis=0)  # Coordinates of the polyline

    if as_geom:
        welded_section = LineString(welded_section)

    return welded_section


def _get_subsect_list(polyline_arr, unit_length):
    subsections_ = []
    i = 0
    j = unit_length + 1
    while i < len(polyline_arr) - 1:
        sub_ls = LineString(polyline_arr[i:i + j])
        if i < len(polyline_arr) - j:
            k = 1
            while unit_length - sub_ls.length >= 0.5:
                sub_ls = LineString(polyline_arr[i:i + j + k])
                k += 1
            i += k - 1
        subsections_.append(sub_ls)
        i += unit_length

    if subsections_[-1].length / unit_length <= 0.4:
        subsections_[-2] = shapely.ops.linemerge(subsections_[-2:])
        del subsections_[-1]

    return subsections_


def split_section(section, unit_length=1, coarsely=False, use_original_points=True, to_geoms=False):
    """
    Split a track section into a number of subsections,
    with each having approximately equal lengths.

    :param section: Track section (represented as a polyline).
    :type section: numpy.ndarray | LineString
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

    .. seealso::

        - Examples for the method
          :meth:`TrackMovement.split_section()<src.shaft.movement.TrackMovement.split_section>`.
    """

    if isinstance(section, np.ndarray):
        polyline = LineString(section)
    else:
        polyline = section

    if unit_length > polyline.length:
        raise InvalidSubsectionLength("`unit_length` should not exceed the length of `section`.")

    else:
        subsect_len = int(np.ceil(polyline.length / unit_length))

        if subsect_len == 1:
            subsections = GeometryCollection([polyline])

        else:
            polyline_arr = np.array(polyline.coords)

            if coarsely:
                subsections_ = _get_subsect_list(polyline_arr, unit_length)

            else:
                splitters = [
                    polyline.interpolate((i / subsect_len), normalized=True)
                    for i in range(1, subsect_len)]
                splitters_arr = np.array(LineString(splitters).coords)

                if use_original_points:  # Use original points
                    polyline_ckdtree = cKDTree(polyline_arr)
                    # distances and indices
                    _, indices = polyline_ckdtree.query(splitters_arr, k=1)
                    indices_ = np.unique(
                        np.insert(np.append(indices, len(polyline_arr) - 1), 0, 0))

                    subsections_ = [
                        LineString(polyline_arr[i:j + 1]) for i, j in paired_next(indices_)]

                else:  # Use the interpolated points
                    splitters_ = np.vstack([polyline_arr[0], splitters_arr, polyline_arr[-1]])
                    subsections_ = [LineString(x) for x in paired_next(splitters_)]

            subsections = GeometryCollection(subsections_)

            if to_geoms:
                subsections = subsections.geoms

        return subsections


def rearrange_line_points(ls1, ls2, ref=None, side='left'):
    """
    Rearrange points of the given lines to adjust the direction of one to the other.

    :param ls1: One line.
    :type ls1: LineString
    :param ls2: Another line.
    :type ls2: LineString
    :param ref: A reference line by which the direction of the given lines are determined;
        when ``ref=None`` (default), the function uses ``ls1`` as the reference.
    :type ref: LineString | None
    :param side: The ``side`` parameter used by
        `shapely.geometry.LineString.parallel_offset()`_; defaults to ``'left'``.
    :type side: str
    :return: Two lines with rearranged points against each other.
    :rtype: tuple

    .. _`shapely.geometry.LineString.parallel_offset()`:
        https://shapely.readthedocs.io/en/stable/manual.html#object.parallel_offset

    .. seealso::

        - Examples for the method
          :meth:`~src.shaft.movement.TrackMovement.calculate_section_movement`.
    """

    # a = LineString(ls1.boundary.geoms).parallel_offset(distance=10, side=side)
    # b = LineString(ls2.boundary.geoms).parallel_offset(distance=10, side=side)
    a, b = map(
        lambda x: LineString(x.boundary.geoms).parallel_offset(distance=10, side=side),
        (ls1, ls2))

    ls1_, ls2_ = map(copy.copy, (ls1, ls2))
    if a.distance(b) - ls1.distance(ls2) > 1:
        if ref is None:
            ls2_ = LineString(ls2.coords[::-1])
        else:
            ls1_ = LineString(ls1.coords[::-1])

    return ls1_, ls2_


def calculate_unit_subsection_displacement(unit_sect_orig, unit_sect_shifted, ref_obj):
    """
    Calculate the displacement of a (short) section (of ~one metre)
    between two different times.

    :param unit_sect_orig: A unit track section's position at an earlier time.
    :type unit_sect_orig: numpy.ndarray | LineString
    :param unit_sect_shifted: (Almost) the same unit track section' position at a later time.
    :type unit_sect_shifted: numpy.ndarray | LineString
    :param ref_obj: A reference object for identifying direction of the displacement.
    :type ref_obj: numpy.ndarray | LineString
    :return: Both lateral and vertical displacements.
    :rtype: tuple
    """

    # unit_sect_orig, unit_sect_shifted, ref_obj = ult201910_1m, ult202004_1m, ref_objects

    if isinstance(ref_obj, np.ndarray):
        reference_xyz = LineString(ref_obj)
    else:
        reference_xyz = ref_obj

    if isinstance(unit_sect_orig, np.ndarray):
        original_xyz = LineString(unit_sect_orig)
    else:
        original_xyz = copy.copy(unit_sect_orig)
    if isinstance(unit_sect_shifted, np.ndarray):
        shifted_xyz = LineString(unit_sect_shifted)
    else:
        shifted_xyz = copy.copy(unit_sect_shifted)

    original_xyz, shifted_xyz = rearrange_line_points(original_xyz, shifted_xyz)

    # -- Lateral displacement ------------------------------------------------------------------
    lateral_displacements = []

    # Centroid of (the original position of) the track section
    original_c_xyz = original_xyz.interpolate(0.5, normalized=True)

    # Define a reference point (on the centre line between the left and right rails)
    _, reference_xy = shapely.ops.nearest_points(original_c_xyz, reference_xyz)

    # Lateral displacement - of the centroid and each of the two ends - of the track section
    shifted_xy = shapely.ops.transform(drop_z, shifted_xyz)

    # Make a collection of key points: [one end, the other end, centroid]
    key_points = original_xyz.boundary.union(original_c_xyz)

    mapping_points = []
    for original_p_xyz in key_points.geoms:
        original_p_xy = shapely.ops.transform(drop_z, original_p_xyz)

        # if xy_shifted.project(original_p_xy) == 0.0:
        _, shifted_p_xy = point_projected_to_line(original_p_xy, shifted_xy)
        # else:  # on the R^2 plane
        #     _, p_xy_shifted = shapely.ops.nearest_points(original_p_xy, xy_shifted)

        p_lateral_displacement = original_p_xy.distance(shifted_p_xy)

        mapping_points.append(Point(np.append(np.array(shifted_p_xy.coords), original_p_xyz.z)))

        # Direction of the displacement
        if original_p_xy.distance(reference_xy) > shifted_p_xy.distance(reference_xy):
            p_lateral_displacement *= -1  # Negative sign means shifting towards the centre

        lateral_displacements.append(p_lateral_displacement)

    # -- Vertical displacement -----------------------------------------------------------------
    vertical_displacements = []

    xz_shifted = shapely.ops.transform(drop_y, shifted_xyz)

    # Vertical displacement - Centroid and two ends of the track section
    for mapping_p_xyz in mapping_points:
        mapping_p_xz = shapely.ops.transform(drop_y, mapping_p_xyz)

        _, p_xz_shifted = point_projected_to_line(mapping_p_xz, xz_shifted)

        # Vertical displacement
        p_vertical_displacement = mapping_p_xz.distance(p_xz_shifted)

        # Direction of the vertical displacement
        if mapping_p_xz.y > p_xz_shifted.y:
            p_vertical_displacement *= -1  # Negative sign means shifting downwards

        vertical_displacements.append(p_vertical_displacement)

    # Note (wrt. the example of this method):
    # A comparison between alternative planes for calculating vertical displacements
    # (X, Z) ~ drop_y           (Y, Z) ~ drop_x
    # 0.00038435758730571485    0.00038433327544285754
    # 0.0008843501549352274     0.0008842942186782473
    # 0.0013843427222473637     0.0013842551604571396

    abs_min_displacement = unit_sect_orig.distance(unit_sect_shifted)

    displacement_data = [lateral_displacements, vertical_displacements, abs_min_displacement]

    return displacement_data


def calc_unit_subsec_disp(unit_sections, ref_obj):
    """
    Calculate the lateral, vertical, and cartesian displacements between two unit sections.

    :param unit_sections: A tuple containing the original and shifted unit sections.
    :type unit_sections: tuple
    :param ref_obj: Reference object used for displacement calculations.
    :type ref_obj: numpy.ndarray | LineString
    :return: A list containing lateral displacement, vertical displacement, and cartesian distance.
    :rtype: list
    """

    unit_sect_orig, unit_sect_shifted = unit_sections

    lateral_disp, vertical_disp = calculate_unit_subsection_displacement(
        unit_sect_orig=unit_sect_orig, unit_sect_shifted=unit_sect_shifted, ref_obj=ref_obj)

    cartesian_distance = unit_sect_orig.distance(unit_sect_shifted)

    disp_dat = [lateral_disp, vertical_disp, cartesian_distance]

    return disp_dat


def get_linestring_ends(ls):
    """
    Get the two ends of a line.

    :param ls: A line.
    :type ls: LineString
    :return: Two points representing the two ends of ``ls``.
    :rtype: tuple
    """

    if ls.length == 0:
        a, b = map(Point, ls.coords)
    else:
        a, b = ls.boundary.geoms

    return a, b


def sort_line_points(line_geoms, as_geom=True):
    """
    Sort a series of LineStrings of the tile-based classified KRDZ data to form a line.

    :param line_geoms: A series of LineStrings of the tile-based classified KRDZ data.
    :type line_geoms: pandas.Series
    :param as_geom: Whether to return as LineString type;
        when ``as_geom=False``, the method returns an array; defaults to ``True``.
    :type as_geom: bool
    :return: An array of continuous line points or an object of ``LineString``.
    :rtype: np.ndarray | LineString
    """

    line_strings = line_geoms.copy()

    i = 0
    while i < len(line_strings) - 1:
        ls1, ls2 = line_strings[i:i + 2]
        (ls1_a, ls1_b), (ls2_a, ls2_b) = map(get_linestring_ends, line_strings[i:i + 2])

        if np.round(ls1_a.distance(ls2_b)) == 1.0:
            ls1_, ls2_ = map(lambda x: LineString(x.coords[::-1]), (ls1, ls2))
            line_strings[i], line_strings[i + 1] = ls1_, ls2_
        elif np.round(ls1_a.distance(ls2_a)) == 1.0:
            line_strings[i] = LineString(ls1.coords[::-1])
        elif np.round(ls1_b.distance(ls2_b)) == 1.0:
            line_strings[i + 1] = LineString(ls2.coords[::-1])

        i += 1

    line_xyz = np.array(line_strings[0].coords)
    for i in range(1, len(line_strings)):
        line_xyz = np.vstack([line_xyz, np.array(line_strings[i].coords)])

    _, idx = np.unique(line_xyz, return_index=True, axis=0)

    line_xyz_sorted = line_xyz[np.sort(idx)]

    # fig = plt.figure(constrained_layout=True)
    # ax = fig.add_subplot(aspect='equal', adjustable='box')
    # ax.plot(xyz[:, 0], xyz[:, 1])

    if as_geom:
        line_xyz_sorted = LineString(line_xyz_sorted)

    return line_xyz_sorted
