"""
This module provides utility tools for general use.
"""

import ast
import copy
import datetime
import functools
import itertools
import json
import os
import pkgutil
import re
import shutil

import folium
import numba
import numpy as np
from pyhelpers._cache import _print_failure_msg
from pyhelpers.dbms import PostgreSQL
from pyhelpers.dirs import cd
from pyhelpers.store import _check_saving_path, save_figure
from pyhelpers.text import cosine_similarity_between_texts, find_similar_str


class TrackFixityDB(PostgreSQL):
    """
    A class inheriting from `pyhelpers.dbms.PostgreSQL
    <https://pyhelpers.readthedocs.io/en/latest/_generated/pyhelpers.dbms.PostgreSQL.html>`_
    as a basic instance of `PostgreSQL <https://www.postgresql.org/>`_
    for managing data of the project.
    """

    def __init__(self, host=None, port=None, username=None, password=None,
                 database_name='NR_TrackFixity', **kwargs):
        """
        :param host: The database host; defaults to ``None``.
        :type host: str | None
        :param port: The database port; defaults to ``None``.
        :type port: int | None
        :param username: The database username; defaults to ``None``.
        :type username: str | None
        :param password: The database password; defaults to ``None``.
        :type password: str | int | None
        :param database_name: The name of the database; defaults to ``NR_TrackFixity``.
        :type database_name: str
        :param kwargs: [Optional] parameters of the class `pyhelpers.dbms.PostgreSQL`_.

        .. _`pyhelpers.dbms.PostgreSQL`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/pyhelpers.dbms.PostgreSQL.html

        **Examples**::

            >>> from src.utils import TrackFixityDB
            >>> db_instance = TrackFixityDB(host='localhost')  # Connect the default local server
            Password (postgres@localhost:5432): ***
            Connecting postgres:***@localhost:5432/NR_TrackFixity ... Successfully.
            >>> db_instance.database_name
            'NR_TrackFixity'
            >>> db_instance = TrackFixityDB()  # Remote server
            >>> db_instance.database_name
            'NR_TrackFixity'
        """

        credentials = {
            'host': host,
            'port': port,
            'username': username,
            'password': password,
            'database_name': database_name,
        }

        if host not in {'localhost', '127.0.0.1'}:
            try:  # Load credentials from the .credentials file
                credentials = json.loads(
                    pkgutil.get_data(__name__, "../data/.credentials").decode())
            except (FileNotFoundError, json.JSONDecodeError):
                pass

        kwargs.update(credentials)
        super().__init__(**kwargs)


def cd_docs_source(*sub_dir, mkdir=False, **kwargs):
    """
    Get path to ``data_dir`` and/or subdirectories / files.

    :param sub_dir: Name of directory or names of directories (and/or a filename).
    :type sub_dir: str
    :param mkdir: Whether to create a directory; defaults to ``False``.
    :type mkdir: bool
    :param kwargs: [Optional] parameters of `os.makedirs`_, e.g. ``mode=0o777``.
    :return path: An absolute path to a directory (or a file) under ``data_dir``.
    :rtype: str

    .. _`os.makedirs`: https://docs.python.org/3/library/os.html#os.makedirs

    **Examples**::

        >>> from src.utils import cd_docs_source
        >>> import os
        >>> path_to_docs_source = cd_docs_source()
        >>> os.path.relpath(path_to_docs_source)
        'docs\\source'
        >>> path_to_docs_img = cd_docs_source("_images")
        >>> os.path.relpath(path_to_docs_img)
        'docs\\source\\_images'
    """

    path = cd("docs\\source", *sub_dir, mkdir=mkdir, **kwargs)

    return path


def eval_data_type(str_val):
    """
    Convert a string to its intrinsic data type.

    :param str_val: A string-type variable.
    :type str_val: str
    :return: Converted value.
    :rtype: typing.Any

    **Examples**::

        >>> from src.utils import eval_data_type
        >>> val_1 = '1'
        >>> origin_val = eval_data_type(val_1)
        >>> type(origin_val)
        int
        >>> origin_val
        1
        >>> val_2 = '1.1.1'
        >>> origin_val = eval_data_type(val_2)
        >>> type(origin_val)
        str
        >>> origin_val
        '1.1.1'
    """

    try:
        val = ast.literal_eval(str_val)
    except (ValueError, SyntaxError):
        val = str_val

    return val


def year_month_to_data_date(year, month, fmt='%Y%m'):
    """
    Convert a pair of ``year`` and ``month`` to a formatted date.

    :param year: Year of when data was collected.
    :type year: str | int
    :param month: Month of when data was collected.
    :type month: str | int
    :param fmt: Format; defaults to ``'%Y%m'``.
    :type fmt: str
    :return: Formatted date of the data.
    :rtype: str

    **Examples**::

        >>> from src.utils import year_month_to_data_date
        >>> year_month_to_data_date(year=2019, month=10)
        '201910'
        >>> year_month_to_data_date(year=2020, month=4)
        '202004'
    """

    data_date = datetime.datetime.strptime('{}{}'.format(year, month), '%Y%m').strftime(fmt)

    return data_date


def data_date_to_year_month(data_date, fmt='%Y%m'):
    """
    Convert a formatted date to a pair of ``year`` and ``month``.

    :param data_date: Date of when the data was collected.
    :type data_date: str | datetime.datetime | datetime.date | int
    :param fmt: Format of the date string; defaults to ``'%Y%m'``.
    :type fmt: str
    :return: Year and month of when data was collected.
    :rtype: tuple

    **Examples**::

        >>> from src.utils import data_date_to_year_month
        >>> data_date_to_year_month(data_date='201910')
        (2019, 10)
        >>> data_date_to_year_month(data_date=10, fmt='%m')
        (None, 10)
        >>> data_date_to_year_month(data_date=20, fmt='%y')
        (2020, None)
    """

    if isinstance(data_date, (datetime.datetime, datetime.date)):
        year, month = data_date.year, data_date.month

    else:
        data_date_ = datetime.datetime.strptime(str(data_date), fmt)

        if fmt in ('%Y', '%y'):
            year, month = data_date_.year, None
        elif fmt in ('%b', '%B', '%m'):
            year, month = None, data_date_.month
        else:
            year, month = data_date_.year, data_date_.month

    return year, month


def add_sql_query_date_condition(sql_query, data_date, date_fmt='%Y%m'):
    """
    Add a condition about data date to a given SQL query statement.

    :param sql_query: SQL query statement.
    :type sql_query: str
    :param data_date: Date of data.
    :type data_date: str | datetime.datetime | datetime.date | int | list | tuple
    :param date_fmt: Format of the date string; defaults to ``'%Y%m'``.
    :type date_fmt: str
    :return: Updated SQL query statement with conditions about data date.
    :rtype: str

    **Examples**::

        >>> from src.utils import add_sql_query_date_condition
        >>> query1 = 'SELECT * FROM a_table'
        >>> query1
        'SELECT * FROM a_table'
        >>> add_sql_query_date_condition(query1, data_date=201910)
        'SELECT * FROM a_table WHERE "Year"=2019 AND "Month"=10'
        >>> add_sql_query_date_condition(query1, data_date='202004')
        'SELECT * FROM a_table WHERE "Year"=2020 AND "Month"=4'
        >>> query2 = 'SELECT * FROM X where a=b'
        >>> query2
        'SELECT * FROM X where a=b'
        >>> add_sql_query_date_condition(query2, data_date=['201910', '202004'])
        'SELECT * FROM X where a=b AND "Year" IN (2019, 2020) AND "Month" IN (10, 4)'
    """

    cond_ = 'AND' if re.search(r'where', sql_query, re.IGNORECASE) else 'WHERE'

    if isinstance(data_date, (str, int)):
        year, month = data_date_to_year_month(data_date=data_date, fmt=date_fmt)
        sql_query += f' {cond_} "Year"={year} AND "Month"={month}'

    elif isinstance(data_date, (list, tuple)):
        year_month = np.array(
            [data_date_to_year_month(data_date=x, fmt=date_fmt) for x in data_date])
        if len(year_month) == 1:
            year, month = year_month[0].tolist()
        else:
            year, month = tuple(year_month[:, 0].tolist()), tuple(year_month[:, 1].tolist())
        sql_query += f' {cond_} "Year" IN {year} AND "Month" IN {month}'

    return sql_query


def get_tile_xy(tile_xy):
    """
    Get X and Y coordinates of a tile.

    :param tile_xy: X and Y coordinates in reference to a tile for the point cloud data.
    :type tile_xy: tuple | list | str
    :return: X and Y coordinates of a tile.
    :rtype: tuple

    **Examples**::

        >>> from src.utils import get_tile_xy
        >>> tile_x, tile_y = get_tile_xy(tile_xy="Tile_X+0000340500_Y+0000674200.laz")
        >>> tile_x, tile_y
        (340500, 674200)
        >>> tile_x, tile_y = get_tile_xy(tile_xy=(340500, 674200))
        >>> tile_x, tile_y
        (340500, 674200)
        >>> tile_x, tile_y = get_tile_xy(tile_xy='X340500_Y674200')
        >>> tile_x, tile_y
        (340500, 674200)
        >>> tile_x, tile_y = get_tile_xy(tile_xy=(340500, None))
        >>> # tile_x, tile_y = get_tile_xy(tile_xy=(340500, ''))
        >>> tile_x, tile_y
        (340500, None)
    """

    if tile_xy is None:
        x, y = None, None

    elif hasattr(tile_xy, '__iter__') and not isinstance(tile_xy, str):

        try:
            x, y = (int(z) if z else None for z in tile_xy)
        except TypeError:
            x, y = map(tuple, np.array(tile_xy).T.tolist())

    else:
        if tile_xy.startswith("Tile_") or tile_xy.endswith(".laz"):
            x, y = (int(z) if z else None for z in re.findall(r'0+\d+', tile_xy))
        else:
            x, y = (int(z) if z else None for z in re.findall(r'\d{6}', tile_xy))

    return x, y


def add_sql_query_xy_condition(sql_query, tile_xy=None):
    """
    Add a condition about the tiles of point cloud data to a given SQL query statement.

    :param sql_query: SQL query statement.
    :type sql_query: str
    :param tile_xy: Easting (X) and northing (Y) of the geographic Cartesian coordinates for a tile;
        defaults to ``None``.
    :type tile_xy: tuple | list | str | None
    :return: Updated SQL query statement with the tiles of point cloud data.
    :rtype: str

    **Examples**::

        >>> from src.utils import add_sql_query_xy_condition
        >>> query = 'SELECT * FROM a_table'
        >>> query
        'SELECT * FROM a_table'
        >>> add_sql_query_xy_condition(query, tile_xy=(340500, 674200))
        'SELECT * FROM a_table WHERE "Tile_X"=340500 AND "Tile_Y"=674200'
        >>> add_sql_query_xy_condition(query, tile_xy=(340500, None))
        'SELECT * FROM a_table WHERE "Tile_X"=340500'
    """

    tile_x, tile_y = get_tile_xy(tile_xy=tile_xy)

    if isinstance(tile_x, int):
        tile_x_ = f' "Tile_X"={tile_x}'
    elif isinstance(tile_x, (tuple, list)):
        tile_x_ = f' "Tile_X" IN {tuple(x for x in tile_x if x is not None)}'
    else:
        tile_x_ = ''

    if tile_x_:
        # sql_query += (f' {"AND" if re.search(r"where", sql_query, re.IGNORECASE) else "WHERE"}'
        #               f'{tile_x_}')
        if re.search(r"where", sql_query, re.IGNORECASE):
            sql_query += f' AND{tile_x_}'
        else:
            sql_query += f' WHERE{tile_x_}'

    if isinstance(tile_y, int):
        tile_y_ = f' "Tile_Y"={tile_y}'
    elif isinstance(tile_y, (tuple, list)):
        tile_y_ = f' "Tile_Y" IN {tuple(y for y in tile_y if y is not None)}'
    else:
        tile_y_ = ''

    if tile_y_:
        sql_query += f' {"AND" if re.search(r"where", sql_query, re.IGNORECASE) else "WHERE"}{tile_y_}'

    return sql_query


def abs_min(x):
    """
    Get the minimum from a sequence of numbers based on their absolute values.

    :param x: A sequence of numbers.
    :type x: typing.Iterable
    :return: The minimum of ``x`` given absolute values.
    :rtype: int | float
    """
    return min(x, key=abs)


def abs_max(x):
    """
    Get the maximum from a sequence of numbers based on their absolute values.

    :param x: A sequence of numbers.
    :type x: typing.Iterable
    :return: The maximum of ``x`` given absolute values.
    :rtype: int | float
    """
    return max(x, key=abs)


def iterable_to_range(iterable):
    """
    Convert an iterable of sequential integers to a range or ranges.

    :param iterable: An iterable of sequential integers.
    :type iterable: typing.Iterable
    :return: A range or ranges.
    :rtype: generator

    **Examples**::

        >>> from src.utils import iterable_to_range
        >>> lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> rng = iterable_to_range(lst)
        >>> type(rng)
        generator
        >>> list(rng)
        [(1, 9)]
    """

    iterable = sorted(set(iterable))

    for key, group in itertools.groupby(enumerate(iterable), lambda x: x[1] - x[0]):
        group = list(group)
        yield group[0][1], group[-1][1]


def paired_next(iterable):
    """
    Loop through an iterable for every two neighbouring elements.

    :param iterable: An iterable variable
    :type iterable: typing.Iterable | numpy.ndarray
    :return: iterator that pairs every two neighbouring elements in the ``iterable``
    :rtype: zip

    **Examples**::

        >>> from src.utils import paired_next
        >>> lst = [1, 3, 5, 7, 9]
        >>> list(paired_next(lst))
        [(1, 3), (3, 5), (5, 7), (7, 9)]
    """

    a, b = itertools.tee(iterable)

    next(b, None)

    return zip(a, b)


def grouped(iterable, n):
    """
    Group every two elements of an iterable variable.

    :param iterable: An iterable variable.
    :param iterable: typing.Iterable
    :param n: Number of elements to be grouped.
    :type n: int
    :return: An iterator that groups every ``n`` elements of the ``iterable``.
    :rtype: zip

    **Examples**::

        >>> from src.utils import grouped
        >>> rng = range(0, 20, 2)
        >>> list(grouped(rng, 2))
        [(0, 2), (4, 6), (8, 10), (12, 14), (16, 18)]
    """

    return zip(*[iter(iterable)] * n)


def add_sql_query_elr_condition(sql_query, elr=None):
    """
    Add a condition about specific ELR(s) to a given SQL query statement.

    :param sql_query: SQL query statement.
    :type sql_query: str
    :param elr: ELR; defaults to ``None``.
    :type elr: tuple | list | str | None
    :return: Updated SQL query statement with specific ELR(s).
    :rtype: str

    **Examples**::

        >>> from src.utils import add_sql_query_elr_condition
        >>> query = 'SELECT * FROM a_table'
        >>> query
        'SELECT * FROM a_table'
        >>> add_sql_query_elr_condition(query)
        'SELECT * FROM a_table'
        >>> add_sql_query_elr_condition(query, elr='ECM8')
        'SELECT * FROM a_table WHERE "ELR"=\'ECM8\''
        >>> add_sql_query_elr_condition(query, elr=['ECM7', 'ECM8'])
        'SELECT * FROM a_table WHERE "ELR" IN (\'ECM7\', \'ECM8\')'
    """

    if isinstance(elr, str):
        elr_ = f' "ELR"=\'{elr}\''
    elif isinstance(elr, (tuple, list)):
        elr_ = f' "ELR" IN {tuple(x for x in elr if x is not None)}'
    else:
        elr_ = ''

    if elr_:
        sql_query += f' {"AND" if re.search(r"where", sql_query, re.IGNORECASE) else "WHERE"}{elr_}'

    return sql_query


@numba.njit
def numba_np_shift(array, step, fill_value=np.nan):
    """
    Shift an array by desired number of rows with Numba.

    :param array: An array of numbers.
    :type array: numpy.ndarray
    :param step: Number of rows to shift.
    :type step: int
    :param fill_value: Values to fill missing rows due to the shift; defaults to ``NaN``.
    :type fill_value: float | int
    :return: Shifted array.
    :rtype: numpy.ndarray

    **Examples**::

        >>> from src.utils import numba_np_shift
        >>> import numpy
        >>> arr = numpy.array([[10, 13, 17],
        ...                    [20, 23, 27],
        ...                    [15, 18, 22],
        ...                    [30, 33, 37],
        ...                    [45, 48, 52]], dtype='float32')
        >>> numba_np_shift(arr, step=-1)
        array([[20., 23., 27.],
               [15., 18., 22.],
               [30., 33., 37.],
               [45., 48., 52.],
               [nan, nan, nan]], dtype=float32)
        >>> numba_np_shift(arr, step=1, fill_value=0)
        array([[ 0.,  0.,  0.],
               [10., 13., 17.],
               [20., 23., 27.],
               [15., 18., 22.],
               [30., 33., 37.]], dtype=float32)
    """

    result = np.empty_like(array)

    if step > 0:
        result[:step] = fill_value
        result[step:] = array[:-step]
    elif step < 0:
        result[step:] = fill_value
        result[:step] = array[-step:]
    else:
        result[:] = array

    return result


def validate_column_names(column_names=None):
    """
    Validate column names for PostgreSQL query statement.

    :param column_names: Column name(s) for a dataframe.
    :type column_names: str | list | tuple | None
    :return: Column names for PostgreSQL query statement.
    :rtype: str

    **Examples**::

        >>> from src.utils import validate_column_names
        >>> validate_column_names()
        '*'
        >>> validate_column_names('col_name_1')
        '"col_name_1"'
        >>> validate_column_names(['col_name_1', 'col_name_2'])
        '"col_name_1", "col_name_2"'
    """

    if isinstance(column_names, str):
        column_names_ = f'"{column_names}"'

    elif isinstance(column_names, (list, tuple)):
        column_names_ = '"{}"'.format('", "'.join([str(col) for col in column_names]))

    else:
        column_names_ = '*'

    return column_names_


def _find_similar_str(x, valid_names):
    y = find_similar_str(x.title(), valid_names)

    if y is None:
        y = max(valid_names, key=functools.partial(cosine_similarity_between_texts, x))

    return y.replace(' ', '')


def find_valid_names(name, valid_names):
    # noinspection PyShadowingNames
    """
    Find valid names.

    :param name: One or a sequence of str values.
    :type name: str | list | None
    :param valid_names: A list of valid str values.
    :type valid_names: list
    :return: Valid value(s).
    :rtype: list

    **Examples:**

        >>> from src.utils import find_valid_names
        >>> valid_names = [
        ...     'LeftTopOfRail', 'LeftRunningEdge', 'RightTopOfRail', 'RightRunningEdge', 'Centre']
        >>> find_valid_names('top left', valid_names)
        ['LeftTopOfRail']
        >>> find_valid_names('right edge', valid_names)
        ['RightRunningEdge']
    """

    if name is not None:
        valid_names_ = [' '.join(re.findall('[A-Z][^A-Z]*', x)) for x in valid_names]
        name_ = [
            _find_similar_str(x, valid_names_)
            for x in ([name] if isinstance(name, str) else name.copy())]

    else:
        name_ = copy.copy(valid_names)

    return name_


def fix_folium_float_image(path_to_m):
    """
    Fix incorrect display of floating images (when utilising folium<=0.13.0).

    :param path_to_m: Path to an HTML file saved by folium.
    :type path_to_m: str
    """

    if folium.__version__ < '0.14.0':
        import bs4

        with open(path_to_m, mode='r', encoding='utf-8') as f:
            soup = bs4.BeautifulSoup(markup=f.read(), features='html.parser')

            float_img_style = [
                x for x in soup.find_all('style') if x.string.strip().startswith('#float_image')][0]

            original_str = float_img_style.string.replace('22px%', '22px')
            _ = float_img_style.string.replace_with(original_str)

        with open(path_to_m, mode='w', encoding='utf-8') as f:
            f.write(str(soup))


def display_folium_m(m, element, direction, subsect_len):
    """
    Display the map produced by folium.

    :param m: Map object created by using folium.
    :param element: Element of rail head, e.g. left/right top of rail or running edge;
        defaults to ``None``.
    :type element: str | list | None
    :param direction: Railway direction, e.g. up and down directions; defaults to ``None``.
    :type direction: str | list | None
    :param subsect_len: Length (in metre) of a subsection for which movement is calculated;
        defaults to ``10``.
    :type subsect_len: int
    :return: Pathname of the output HTML file.
    :rtype: str
    """

    html_filename = f"{direction}_{element}_{subsect_len}.html".lower().replace(" ", "_")
    html_pathname = cd(f"demos\\heatmaps\\{html_filename}", mkdir=True)

    m.save(html_pathname)

    fix_folium_float_image(html_pathname)

    iframe_args = dict(src=f"heatmaps\\{html_filename}", width=975, height=700)

    import IPython.display
    IPython.display.IFrame(**iframe_args)

    return iframe_args


def save_plot(fig, filename, save_as, path_to_img_dir=None, verbose=False, **kwargs):
    """
    Save the given plot in multiple formats.

    :param fig: The figure object to save.
    :type fig: matplotlib.figure.Figure
    :param filename: The base filename for the saved plot (without extension).
    :type filename: str
    :param save_as: A string or a list of strings indicating the file format(s) to save the plot as.
    :type save_as: str | list[str]
    :param path_to_img_dir: The directory to save the plot files; defaults to the current directory;
        defaults to ``None``.
    :type path_to_img_dir: str | None
    :param verbose: Whether to print relevant information to the console; defaults to ``False``
    :type verbose: bool | int
    :param kwargs: [Optional] parameters of the `pyhelpers.store.save_figure`_ function.

    .. _`pyhelpers.store.save_figure`:
        https://pyhelpers.readthedocs.io/en/latest/_generated/pyhelpers.store.save_figure.html
    """

    save_as_ = {save_as} if isinstance(save_as, str) else set(save_as)
    valid_file_ext = {
        "eps", "jpeg", "jpg", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz", "tif",
        "tiff", "webp"}

    # Check if all extensions in save_as are valid
    invalid_ext = [x for x in save_as_ if x.lstrip('.') not in valid_file_ext]
    if invalid_ext:
        raise ValueError(
            f"Invalid file extension(s) in `save_as`: {invalid_ext}. "
            f"Must be chosen from {valid_file_ext}.")

    save_as_.add(".pdf")

    for ext in save_as_:
        fig_filename = f"{filename}.{ext.lstrip('.')}"
        fig_pathname = cd(path_to_img_dir, fig_filename, mkdir=True)
        save_figure(fig, fig_pathname, verbose=verbose, **kwargs)

        docs_img_path = cd("docs\\source\\_images", fig_filename, mkdir=True)

        if fig_pathname != docs_img_path:
            _check_saving_path(docs_img_path, verbose=verbose, state_verb="Copying")
            try:
                shutil.copy2(fig_pathname, docs_img_path)
                if os.path.exists(docs_img_path) and verbose:
                    print("Done.")
            except Exception as e:
                _print_failure_msg(e)
