"""Test the module :mod:`~src.utils`."""

import importlib.resources
import typing

import pytest
from shapely.geometry import LineString, Point

from src.utils.dgn import *
from src.utils.general import *
from src.utils.geometry import *
from src.utils.las import *


def test_cd_docs_source():
    path_to_docs_source = cd_docs_source()
    assert os.path.relpath(path_to_docs_source) == 'docs\\source'

    path_to_docs_img = cd_docs_source("_images")
    assert os.path.relpath(path_to_docs_img) == 'docs\\source\\_images'


def test_eval_data_type():
    val_1 = '1'
    origin_val = eval_data_type(val_1)
    assert isinstance(origin_val, int)
    assert origin_val == 1

    val_2 = '1.1.1'
    origin_val = eval_data_type(val_2)
    assert isinstance(origin_val, str)
    assert origin_val == '1.1.1'


def test_year_month_to_data_date():
    rslt = year_month_to_data_date(year=2019, month=10)
    assert rslt == '201910'
    rslt = year_month_to_data_date(year=2020, month=4)
    assert rslt == '202004'


def test_data_date_to_year_month():
    rslt = data_date_to_year_month(data_date='201910')
    assert rslt == (2019, 10)
    rslt = data_date_to_year_month(data_date=10, fmt='%m')
    assert rslt == (None, 10)
    rslt = data_date_to_year_month(data_date=20, fmt='%y')
    assert rslt == (2020, None)


def test_add_sql_query_date_condition():
    query1 = 'SELECT * FROM a_table'
    query1_ = add_sql_query_date_condition(query1, data_date=201910)
    assert query1_ == 'SELECT * FROM a_table WHERE "Year"=2019 AND "Month"=10'

    query1_ = add_sql_query_date_condition(query1, data_date='202004')
    assert query1_ == 'SELECT * FROM a_table WHERE "Year"=2020 AND "Month"=4'

    query2 = 'SELECT * FROM X where a=b'
    query2_ = add_sql_query_date_condition(query2, data_date=['201910', '202004'])
    assert query2_ == 'SELECT * FROM X where a=b AND "Year" IN (2019, 2020) AND "Month" IN (10, 4)'


def test_get_tile_xy():
    tile_x, tile_y = get_tile_xy(tile_xy="Tile_X+0000340500_Y+0000674200.laz")
    assert tile_x, tile_y == (340500, 674200)

    tile_x, tile_y = get_tile_xy(tile_xy=(340500, 674200))
    assert tile_x, tile_y == (340500, 674200)

    tile_x, tile_y = get_tile_xy(tile_xy='X340500_Y674200')
    assert tile_x, tile_y == (340500, 674200)

    tile_x, tile_y = get_tile_xy(tile_xy=(340500, None))
    assert tile_x, tile_y == (340500, None)


def test_add_sql_query_xy_condition():
    query = 'SELECT * FROM a_table'

    query_ = add_sql_query_xy_condition(query, tile_xy=(340500, 674200))
    assert query_ == 'SELECT * FROM a_table WHERE "Tile_X"=340500 AND "Tile_Y"=674200'

    query_ = add_sql_query_xy_condition(query, tile_xy=(340500, None))
    assert query_ == 'SELECT * FROM a_table WHERE "Tile_X"=340500'


def test_iterable_to_range():
    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    rng = iterable_to_range(lst)
    assert isinstance(rng, typing.Generator)
    assert list(rng) == [(1, 9)]


def test_paired_next():
    lst = [1, 3, 5, 7, 9]
    rslt = list(paired_next(lst))
    assert rslt == [(1, 3), (3, 5), (5, 7), (7, 9)]


def test_grouped():
    rng = range(0, 20, 2)
    rslt = list(grouped(rng, 2))
    assert rslt == [(0, 2), (4, 6), (8, 10), (12, 14), (16, 18)]


def test_add_sql_query_elr_condition():
    query = 'SELECT * FROM a_table'

    query_ = add_sql_query_elr_condition(query)
    assert query_ == 'SELECT * FROM a_table'

    query_ = add_sql_query_elr_condition(query, elr='ECM8')
    assert query_ == 'SELECT * FROM a_table WHERE "ELR"=\'ECM8\''

    query_ = add_sql_query_elr_condition(query, elr=['ECM7', 'ECM8'])
    assert query_ == 'SELECT * FROM a_table WHERE "ELR" IN (\'ECM7\', \'ECM8\')'


def test_numba_np_shift():
    arr = np.array([[10, 13, 17],
                    [20, 23, 27],
                    [15, 18, 22],
                    [30, 33, 37],
                    [45, 48, 52]], dtype='float32')

    arr_ = numba_np_shift(arr, step=-1)
    assert np.isnan(arr_[-1, :]).all()

    arr_ = numba_np_shift(arr, step=1, fill_value=0)
    # noinspection PyUnresolvedReferences
    assert (arr_[0, :] == 0).all()


def test_validate_column_names():
    col_names = None
    col_names_ = validate_column_names(col_names)
    assert col_names_ == '*'

    col_names = 'col_name_1'
    col_names_ = validate_column_names(col_names)
    assert col_names_ == '"col_name_1"'

    col_names = ['col_name_1', 'col_name_2']
    col_names_ = validate_column_names(col_names)
    assert col_names_ == '"col_name_1", "col_name_2"'


def test_find_valid_names():
    v_names = [
        'LeftTopOfRail',
        'LeftRunningEdge',
        'RightTopOfRail',
        'RightRunningEdge',
        'Centre',
    ]
    rslt = find_valid_names('top left', v_names)
    assert rslt == ['LeftTopOfRail']

    rslt = find_valid_names('right edge', v_names)
    assert rslt == ['RightRunningEdge']


def test_flatten_geometry():
    path_to_geom_ = importlib.resources.files(__package__).joinpath("data/example_geometry.pkl")
    with importlib.resources.as_file(path_to_geom_) as path_to_geom:
        example_geometry = load_data(path_to_geom)
    assert isinstance(example_geometry, pd.Series) and example_geometry.shape == (5,)
    example_geometry_ = flatten_geometry(example_geometry)
    assert isinstance(example_geometry_, np.ndarray) and example_geometry_.shape == (10, 3)


def test_calculate_slope():
    arr1 = np.array([[399299.5160, 655099.1290], [399299.9990, 655098.2540]])
    slope1 = calculate_slope(arr1)
    assert np.round(slope1, 8) == np.round(-1.8116169544740974, 8)
    slope1_1 = calculate_slope(arr1, method='numpy')
    assert np.round(slope1_1, 8) == np.round(-1.8115942032905608, 8)
    slope1_2 = calculate_slope(arr1, method='scipy')
    assert np.round(slope1_2, 8) == np.round(-1.8115942028706058, 8)

    path_to_arr2_ = importlib.resources.files(__package__).joinpath("data/example_array.pkl")
    with importlib.resources.as_file(path_to_arr2_) as path_to_arr2:
        arr2 = load_data(path_to_arr2)
    slope2 = calculate_slope(arr2)
    assert np.round(slope2, 8) == np.round(-1.807090900484403, 8)
    slope2_1 = calculate_slope(arr2, method='numpy')
    assert np.round(slope2_1, 8) == np.round(-1.8070743791308161, 8)
    slope2_2 = calculate_slope(arr2, method='scipy')
    assert np.round(slope2_2, 8) == np.round(-1.8070743792147357, 8)


def test_point_projected_to_line():
    pt = Point([399297, 655095, 43])
    ls = LineString([[399299, 655091, 42], [399295, 655099, 42]])

    _, pt_ = point_projected_to_line(pt, ls)
    assert pt_.wkt == 'POINT Z (399297 655095 42)'


def test_make_a_polyline():
    polyline_coords = [(360100, 677100), (360000, 677100), (360200, 677100)]

    rslt = make_a_polyline(polyline_coords, start_point=(360000, 677100))
    assert rslt.wkt == 'LINESTRING (360000 677100, 360100 677100, 360200 677100)'


def test_offset_ls():
    x1 = np.arange(start=0, stop=10, step=0.1)
    y1 = x1 + 1

    x2 = np.arange(start=1, stop=20, step=0.5)
    y2 = 2 * x2 + 2

    l1_arr, l2_arr = map(np.column_stack, ((x1, y1), (x2, y2)))
    l1_1, l2_1 = offset_ls(ls1=l1_arr, ls2=l2_arr, as_array=True)

    assert np.array_equal(np.round(l2_1, 8), np.array([[1.00000000, 4.00000000],
                                                       [1.50000000, 5.00000000],
                                                       [2.00000000, 6.00000000],
                                                       [2.50000000, 7.00000000],
                                                       [3.00000000, 8.00000000],
                                                       [3.50000000, 9.00000000],
                                                       [4.00000000, 10.00000000],
                                                       [4.50000000, 11.00000000],
                                                       [5.00000000, 12.00000000],
                                                       [5.50000000, 13.00000000],
                                                       [5.69763121, 13.39526241]]))

    l1_2, l2_2 = offset_ls(ls1=l1_arr, ls2=l2_arr, cap_style=2, as_array=True)
    assert np.array_equal(np.round(l2_2, 8), np.array([[1.00000000, 4.00000000],
                                                       [1.50000000, 5.00000000],
                                                       [2.00000000, 6.00000000],
                                                       [2.50000000, 7.00000000],
                                                       [3.00000000, 8.00000000],
                                                       [3.50000000, 9.00000000],
                                                       [4.00000000, 10.00000000],
                                                       [4.50000000, 11.00000000],
                                                       [5.00000000, 12.00000000],
                                                       [5.50000000, 13.00000000],
                                                       [6.00000000, 14.00000000],
                                                       [6.26666667, 14.53333333]]))


def test_extrapolate_line_point():
    ls_arr = np.array([[326301.3395, 673958.8021, 0.0000],
                       [326302.2164, 673959.0613, 0.0000],
                       [326309.2315, 673961.1349, 0.0000]])
    ls = LineString(ls_arr)

    pt = extrapolate_line_point(ls, dist=1, deg=1, reverse=True)
    assert np.round(pt.x, 8) == np.round(326300.3805232464, 8)
    assert np.round(pt.y, 8) == np.round(673958.5186338174, 8)


if __name__ == '__main__':
    pytest.main()
