"""
This module provides helper classes/functions for dealing with geometric and geospatial data.
"""

import copy
import functools
import itertools
import math

import numpy as np
import shapely.geometry
import shapely.ops
import shapely.wkt
from pyhelpers.geom import find_shortest_path
from scipy.interpolate import interp1d
from scipy.stats import linregress


def flatten_geometry(geom, as_array=True):
    """
    Flatten a series of geometry object.

    :param geom: A series of geometry object.
    :type geom: pandas.Series
    :param as_array: Whether to return the flattened geometry as an array; defaults to ``True``.
    :type as_array: bool
    :return: An array of point coordinates.
    :rtype: list | numpy.ndarray

    **Examples**::

        >>> from src.utils import flatten_geometry, TrackFixityDB
        >>> from src.preprocessor import PCD
        >>> from pyhelpers.settings import np_preferences
        >>> db_instance = TrackFixityDB()
        >>> pcd = PCD(db_instance=db_instance)
        >>> tbl_name = pcd.dgn_shp_table_name_('Polyline')
        >>> query = f'SELECT * FROM "PCD"."{tbl_name}" WHERE "Year"=2020 AND "Month"=4 LIMIT 5'
        >>> example_dat = pcd.db_instance.read_sql_query(query)
        >>> example_dat['geometry']
        0    LINESTRING Z (340183.475 674108.682 33.237, 34...
        1    LINESTRING Z (340184.412 674109.031 33.235, 34...
        2    LINESTRING Z (340185.349 674109.38 33.232, 340...
        3    LINESTRING Z (340233.146 674127.169 33.117, 34...
        4    LINESTRING Z (340234.083 674127.518 33.114, 34...
        Name: geometry, dtype: object
        >>> np_preferences()
        >>> flatten_geometry(geom=example_dat.geometry)
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

    try:
        geom_coords = geom.map(lambda x: list(shapely.wkt.loads(x).coords))
    except TypeError:
        geom_coords = geom.map(lambda x: list(x.coords))

    flat_coords_list = list(itertools.chain.from_iterable(geom_coords))

    if as_array:
        flat_coords_list = np.array(flat_coords_list)

    return flat_coords_list


def calculate_slope(xy_arr, method=None, na_val=None, **kwargs):
    """
    Calculate the slope for a given 2D array.

    :param xy_arr: A 2D array.
    :type xy_arr: numpy.ndarray
    :param method: Method used to calculate the slope, and options include ``{'numpy', 'scipy'}``;
        defaults to ``None``.
    :type method: str | None
    :param na_val: Replace NA/NaN (which is the result) with a specific value; defaults to ``None``.
    :type na_val: numpy.nan | int | float | None
    :return: Slope of ``xy_arr``.
    :rtype: float

    **Examples**::

        >>> from src.utils import calculate_slope
        >>> import numpy
        >>> arr1 = numpy.array([[399299.5160, 655099.1290], [399299.9990, 655098.2540]])
        >>> slope1 = calculate_slope(arr1)
        >>> slope1
        -1.8116169544740974
        >>> slope1_1 = calculate_slope(arr1, method='numpy')
        >>> slope1_1
        -1.8115942032905608
        >>> slope1_2 = calculate_slope(arr1, method='scipy')
        >>> slope1_2
        -1.8115942028706058
        >>> arr2 = numpy.array([[399299.9540, 655091.4080, 43.2010],
        ...                     [399299.4700, 655092.2840, 43.2060],
        ...                     [399298.9850, 655093.1580, 43.2110],
        ...                     [399298.5010, 655094.0330, 43.2170],
        ...                     [399298.0160, 655094.9070, 43.2230],
        ...                     [399297.5330, 655095.7830, 43.2290],
        ...                     [399297.0490, 655096.6580, 43.2330],
        ...                     [399296.5650, 655097.5330, 43.2390],
        ...                     [399296.0800, 655098.4080, 43.2430],
        ...                     [399295.5960, 655099.2830, 43.2480]])
        >>> slope2 = calculate_slope(arr2)
        >>> slope2
        -1.807090900484403
        >>> slope2_1 = calculate_slope(arr2, method='numpy')
        >>> slope2_1
        -1.8070743791308161
        >>> slope2_2 = calculate_slope(arr2, method='scipy')
        >>> slope2_2
        -1.8070743792147357
    """

    if len(xy_arr) > 1:
        x, y = xy_arr[:, 0], xy_arr[:, 1]

        if method == 'numpy':
            slope = np.polyfit(x=x, y=y, deg=1, **kwargs)[0]
        elif method == 'scipy':
            slope = linregress(x=x, y=y).slope
        else:
            slope = ((x * y).mean() - x.mean() * y.mean()) / ((x ** 2).mean() - (x.mean()) ** 2)

    else:
        slope = np.nan if na_val is None else na_val

    return slope


def drop_z(x, y, _):
    return x, y


def drop_y(x, _, z):
    return x, z


def drop_x(_, y, z):
    return y, z


def point_projected_to_line(point, line, drop=None):
    """
    Find the projected point from a known point to a line.

    :param point: Geometry object of a point.
    :type point: shapely.geometry.Point
    :param line: Ggeometry object of a line.
    :type line: shapely.geometry.LineString
    :param drop: Which dimension to drop, and options include {``'x', 'y', 'z'``};
        defaults to ``None``.
    :type drop: str | None
    :return: The original point (with all or partial dimensions, given ``drop``) and
        the projected one.
    :rtype: tuple

    **Examples**::

        >>> from src.utils import point_projected_to_line
        >>> from shapely.geometry import Point, LineString
        >>> pt = Point([399297, 655095, 43])
        >>> ls = LineString([[399299, 655091, 42], [399295, 655099, 42]])
        >>> _, pt_ = point_projected_to_line(pt, ls)
        >>> pt_.wkt
        'POINT Z (399297 655095 42)'
    """

    if line.length == 0:
        point_, line_ = point, shapely.geometry.Point(line.coords)

    else:
        if drop is not None:
            assert drop in ('x', 'y', 'z')
            func = globals()['drop_{}'.format(drop)]
            point_, line_ = shapely.ops.transform(func, point), shapely.ops.transform(func, line)
        else:
            point_, line_ = copy.copy(point), copy.copy(line)

        x = np.array(point_.coords[0])

        u = np.array(line_.coords[0])
        v = np.array(line_.coords[len(line_.coords) - 1])

        n = v - u
        n /= np.linalg.norm(n, 2)

        p = u + n * np.dot(x - u, n)

        line_ = shapely.geometry.Point(p)

    return point_, line_


def calculate_normed_distance_along_path(polyline):
    distance = np.cumsum(np.sqrt(np.sum(np.diff(polyline, axis=1) ** 2, axis=0)))

    return np.insert(distance, 0, 0) / distance[-1]


def calculate_average_distance_between_polylines(xy1, xy2):
    s1, s2 = map(calculate_normed_distance_along_path, (xy1, xy2))

    interpol_xy1 = interp1d(s1, xy1)
    xy1_on_2 = interpol_xy1(s2)

    node_to_node_distance = np.sqrt(np.sum((xy1_on_2 - xy2) ** 2, axis=0))

    return node_to_node_distance.mean()  # or use the max


def make_a_polyline(points_sequence, start_point=None, reverse=False, as_geom=True):
    """
    Sort a sequence of geographic coordinates to form a real path.

    :param points_sequence: A sequence of geographic coordinates.
    :type points_sequence: shapely.geometry.LineString | list | numpy.ndarray
    :param start_point: Start point of a path; defaults to ``None``.
    :type start_point: list | tuple | numpy.ndarray | shapely.geometry.Point | None
    :param reverse: Whether to reverse the resultant path; defaults to ``False``.
    :type reverse: bool
    :param as_geom: Whether to return the sorted path as a line geometry object;
        defaults to ``True``.
    :type as_geom: bool
    :return: (Sorted) coordinates of the input ``polyline``.
    :rtype: list | shapely.geometry.LineString

    **Examples**::

        >>> from src.utils import make_a_polyline
        >>> polyline_coords = [(360100, 677100), (360000, 677100), (360200, 677100)]
        >>> # polyline = polyline_coords.copy()
        >>> # start_point = (360000, 677100)
        >>> rslt = make_a_polyline(polyline_coords, start_point=(360000, 677100))
        >>> rslt.wkt
        'LINESTRING (360000 677100, 360100 677100, 360200 677100)'
    """

    def euclidean_distance(p1, p2):
        dist = [(a - b) ** 2 for a, b in zip(p1, p2)]
        dist = math.sqrt(sum(dist))
        return dist

    if isinstance(points_sequence, shapely.geometry.LineString):
        polyline_ = list(points_sequence.coords)
    elif isinstance(points_sequence, np.ndarray):
        polyline_ = points_sequence.tolist()
    else:
        polyline_ = points_sequence

    if isinstance(start_point, shapely.geometry.Point):
        start_point_ = [start_point.x, start_point.y]
    elif isinstance(start_point, np.ndarray):
        start_point_ = start_point.tolist()
    # elif isinstance(start_point, tuple):
    #     start_point = list(start_point)
    elif start_point is None:
        start_point_ = polyline_[0]
    else:
        start_point_ = start_point

    if not isinstance(start_point_, type(polyline_[0])):
        start_point_ = type(polyline_[0])(start_point_)

    examined_set = polyline_.copy()
    examined_set.remove(start_point_)

    shortest_path = [start_point_]

    while examined_set:
        nearest_point = min(examined_set, key=lambda x: euclidean_distance(shortest_path[-1], x))
        shortest_path.append(nearest_point)

        examined_set.remove(nearest_point)

    # if int_type:
    #     # list(tuple(int(elem) for elem in lst) for lst, _ in itertools.groupby(shortest_path))
    #     shortest_path = np.array(shortest_path).astype(int).tolist()

    if reverse:
        shortest_path.reverse()

    if as_geom:
        try:
            shortest_path = shapely.geometry.LineString(shortest_path)

        except ValueError:
            from shapely import speedups

            speedups.disable()

            shortest_path = shapely.geometry.LineString(shortest_path)

    return shortest_path


def offset_ls(ls1, ls2, cap_style=1, as_array=False):
    """
    Offset a linestring of longer length (``ls2``) by a shorter one (``ls1``).

    :param ls1: A line.
    :type ls1: shapely.geometry.LineString | numpy.ndarray
    :param ls2: A line (shorter than ``ls1``).
    :type ls2: shapely.geometry.LineString | numpy.ndarray
    :param cap_style: ``cap_style`` enumerated by the object `shapely.geometry.CAP_STYLE`_;
        defaults to ``1``.
    :type cap_style: int
    :param as_array: Whether to return the result as `numpy.ndarray`_ type; defaults to ``False``.
    :type as_array: bool
    :return: ``ls1`` and shortened ``ls2``.
    :rtype: tuple

    .. _`shapely.geometry.CAP_STYLE`:
        https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.CAP_STYLE
    .. _`numpy.ndarray`:
        https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html

    **Examples**::

        >>> from src.utils import offset_ls
        >>> from pyhelpers.settings import np_preferences
        >>> import numpy
        >>> np_preferences()
        >>> x1 = numpy.arange(start=0, stop=10, step=0.1)
        >>> y1 = x1 + 1
        >>> x2 = numpy.arange(start=1, stop=20, step=0.5)
        >>> y2 = 2 * x2 + 2
        >>> l1_arr, l2_arr = map(numpy.column_stack, ((x1, y1), (x2, y2)))
        >>> l1_1, l2_1 = offset_ls(ls1=l1_arr, ls2=l2_arr, as_array=True)
        >>> l2_1
        array([[1.0000, 4.0000],
               [1.5000, 5.0000],
               [2.0000, 6.0000],
               [2.5000, 7.0000],
               [3.0000, 8.0000],
               [3.5000, 9.0000],
               [4.0000, 10.0000],
               [4.5000, 11.0000],
               [5.0000, 12.0000],
               [5.5000, 13.0000],
               [5.6976, 13.3953]])
        >>> l1_2, l2_2 = offset_ls(ls1=l1_arr, ls2=l2_arr, cap_style=2, as_array=True)
        >>> l2_2
        array([[1.0000, 4.0000],
               [1.5000, 5.0000],
               [2.0000, 6.0000],
               [2.5000, 7.0000],
               [3.0000, 8.0000],
               [3.5000, 9.0000],
               [4.0000, 10.0000],
               [4.5000, 11.0000],
               [5.0000, 12.0000],
               [5.5000, 13.0000],
               [6.0000, 14.0000],
               [6.2667, 14.5333]])

    **Illustration**::

        import matplotlib.pyplot as plt
        from pyhelpers.settings import mpl_preferences
        mpl_preferences(backend='TkAgg')
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(x1, y1, label='ls1')
        ax.plot(x2, y2, label='ls2')
        # cap_style=1
        ax.plot(l2_1[:, 0], l2_1[:, 1], label='ls2 (offset; cap_style=1)', linewidth=8)
        # cap_style=2
        ax.plot(l2_2[:, 0], l2_2[:, 1], label='ls2 (offset; cap_style=2)', linewidth=4)
        ax.legend(loc='best')
        fig.tight_layout()
        # fig.savefig("docs\\source\\_images\\utils_offset_ls_demo.svg")
        # fig.savefig("docs\\source\\_images\\utils_offset_ls_demo.pdf")

    .. figure:: ../_images/utils_offset_ls_demo.*
        :name: utils_offset_ls_demo
        :align: center
        :width: 80%

        An example of a line ('ls2') offset  by another ('ls1').
    """

    # Shorter one
    ls1_ = shapely.geometry.LineString(ls1) if isinstance(ls1, np.ndarray) else copy.copy(ls1)
    # Longer one
    ls2_ = shapely.geometry.LineString(ls2) if isinstance(ls2, np.ndarray) else copy.copy(ls2)

    if ls1_.length == 0:
        temp_ = sorted(ls2_.coords, key=lambda x: ls1_.distance(shapely.geometry.Point(x)))
        temp = np.array(temp_[:3])
    else:
        if cap_style == 1:
            xyz2_cen = shapely.ops.nearest_points(ls1_.centroid, ls2_)[1]
            buf = xyz2_cen.buffer(distance=ls1_.length / 2, cap_style=cap_style)
        else:
            buf = ls1_.buffer(distance=ls1_.centroid.distance(ls2_.centroid), cap_style=cap_style)
        temp = np.array(buf.intersection(ls2_).coords)

    ls2_ = find_shortest_path(points_sequence=temp)

    if not as_array:
        ls1_, ls2_ = map(shapely.geometry.LineString, (ls1_, ls2_))

    return ls1_, ls2_


def geom_distance(geom_obj):
    """
    Get a key (function type) to search for relative distance.

    :param geom_obj: A geometry object.
    :type: shapely.geometry.base.BaseGeometry
    :return: The 'key' to search for relative distance.
    :rtype: functools.partial

    .. seealso::

        Examples of the method
        :py:meth:`TrackMovement.calculate_unit_subsection_displacement()
        <src.shaft.TrackMovement.calculate_unit_subsection_displacement>`.
    """

    key = functools.partial(shapely.geometry.base.BaseGeometry.distance, geom_obj)

    return key


def extrapolate_line_point(polyline, dist, deg=1, reverse=False, as_geom=True):
    """
    Extrapolate the coordinates of a point that lies on extension of a given line at a given distance.

    :param polyline: A polynomial line
    :type polyline: shapely.geometry.LineString
    :param dist: distance (in metre) beyond an end of the given line
    :type dist: float | int
    :param deg: degree of the polynomial line; defaults to ``1``
    :type deg: int
    :param reverse: indicate the direction of th extrapolation; defaults to ``False``
    :type reverse: bool
    :param as_geom: Whether to return as a shapely geometry object; defaults to ``True``
    :type as_geom: bool
    :return: A point that extends a given line at a given distance
    :rtype: tuple | shapely.geometry.Point

    **Examples**::

        >>> from src.utils import extrapolate_line_point
        >>> from pyhelpers.settings import np_preferences
        >>> from shapely.geometry import LineString
        >>> import numpy
        >>> np_preferences()
        >>> ls_arr = numpy.array([[326301.3395, 673958.8021, 0.0000],
        ...                       [326302.2164, 673959.0613, 0.0000],
        ...                       [326309.2315, 673961.1349, 0.0000]])
        >>> ls = LineString(ls_arr)
        >>> pt = extrapolate_line_point(ls, dist=1, deg=1, reverse=True)
        >>> pt.wkt
        'POINT (326300.3805232464 673958.5186338174)'

    **Illustration**::

        import matplotlib.pyplot as plt
        from pyhelpers.settings import mpl_preferences
        mpl_preferences(backend='TkAgg')
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.ticklabel_format(useOffset=False)
        ax.plot(ls_arr[:, 0], ls_arr[:, 1], label='A line')
        ax.scatter(pt.x, pt.y, label='The point being extrapolated')
        ax.legend()
        fig.tight_layout()
        # fig.savefig("docs\\source\\_images\\utils_extrapolate_line_point_demo.svg")
        # fig.savefig("docs\\source\\_images\\utils_extrapolate_line_point_demo.pdf")

    .. figure:: ../_images/utils_extrapolate_line_point_demo.*
        :name: utils_extrapolate_line_point_demo
        :align: center
        :width: 80%

        An example of a point being extrapolated from a given line.
    """

    line_arr = np.array(polyline.coords)
    x, y = line_arr[:, 0], line_arr[:, 1]
    line_coeff = np.polyfit(x, y, deg=deg)

    p = np.poly1d(line_coeff)

    a = line_coeff[0]
    b = abs(np.sin(np.arctan(line_coeff[0]))) * dist

    if reverse:
        if x[0] > x[-1]:
            y0 = y[0] - (-1 if a > 0 else 1) * b
        else:
            y0 = y[0] + (-1 if a > 0 else 1) * b
    else:
        if x[0] > x[-1]:
            y0 = y[-1] - (1 if a > 0 else -1) * b
        else:
            y0 = y[-1] + (1 if a > 0 else -1) * b

    x0 = (p - y0).roots[0]

    p0 = shapely.geometry.Point((x0, y0)) if as_geom else (x0, y0)

    return p0
