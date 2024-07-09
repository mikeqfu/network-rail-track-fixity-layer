"""
This module provides helper classes and functions for working with DGN data and converting
shapefiles.

It aims to simplify the process of handling and transforming DGN data formats and shapefiles,
providing a range of utilities for common operations.
"""

import datetime
import glob
import os
import shutil
import webbrowser

import dateutil.parser
import execnet
import folium
import folium.plugins
import numpy as np
import osgeo.osr
import pandas as pd
import pycrs
from pydriosm.reader import SHPReadParse
from pyhelpers.dirs import cd, cdd
from pyhelpers.geom import osgb36_to_wgs84
from pyhelpers.ops import confirmed
from pyhelpers.store import load_data, save_data
from pyhelpers.text import find_similar_str

from src.utils.general import fix_folium_float_image


def parse_projcs(path_to_projcs, parser='osr', as_dict=False):
    """
    Parse a PROJCS (Projected Coordinate Systems) data file.

    This function reads and parses a .prj data file,
    which contains information about projected coordinate systems.

    :param path_to_projcs: The file path to a .prj data file.
    :type path_to_projcs: str
    :param parser: The name of the package used to read the .prj data file.
                   Valid options are ``{'osr', 'pycrs'}``; defaults to ``'osr'``.
    :type parser: str
    :param as_dict: Whether to return the data as a dictionary; defaults to True.
    :type as_dict: bool
    :return: The parsed data from the .prj file, either as a string or a dictionary.
    :rtype: str | dict

    :raises ValueError: If the provided parser is not one of the valid options.
    :raises FileNotFoundError: If the .prj file cannot be found at the given path.
    :raises Exception: For any other errors during file parsing.

    .. seealso::

        :py:func:`~src.utils.get_dgn_shp_prj` for examples of this function in use.
    """

    if parser == 'osr':
        prj_file_ = open(path_to_projcs, 'r').read()

        srs = osgeo.osr.SpatialReference()

        if srs.ImportFromWkt(prj_file_) or srs.ImportFromProj4(prj_file_):
            raise ValueError("Error importing PRJ information from: {}".format(
                os.path.relpath(path_to_projcs)))

        prj_file = srs.ExportToProj4()
        if as_dict:
            prj_file = dict([x.split('=') for x in prj_file.replace('+', '').split(' ')][:-1])

    else:
        prj_file_ = pycrs.load.from_file(path_to_projcs)
        prj_file = prj_file_.to_proj4(as_dict=as_dict)

    return prj_file


def get_dgn_shp_prj(path_to_file, projcs_parser='osr', as_dict=True, update=False, verbose=False):
    """
    Read metadata of projection for DGN files or shapefiles.

    :param path_to_file: Path to a PROJCS file (without the .prj file extension).
    :type path_to_file: str
    :param projcs_parser: Name of the package used to read the .prj data file.
                          Valid options are ``{'osr', 'pycrs'}``; defaults to ``'osr'``.
    :type projcs_parser: str
    :param as_dict: Whether to return the data as a dictionary; defaults to ``True``.
    :type as_dict: bool
    :param update: Whether to read the original data instead of loading from a pickle file
                   if available; defaults to ``False``.
    :type update: bool
    :param verbose: Whether to print relevant information to the console as the function runs;
                    defaults to ``False``.
    :type verbose: bool | int
    :return: Parsed PROJCS data and shape boundary. Returns ``None`` if parsing fails.
    :rtype: dict | None

    :raises ValueError: If the provided parser is not one of the valid options.
    :raises FileNotFoundError: If the .prj file cannot be found at the given path.
    :raises Exception: For any other errors during file parsing.

    **Examples**::

        >>> from src.utils import get_dgn_shp_prj
        >>> from pyhelpers.dirs import cd
        >>> from src.preprocessor import OPAS
        >>> opas = OPAS()
        >>> file_path = cd(opas.DATA_DIR, opas.PROJ_DIRNAME, opas.PROJ_FILENAME)
        >>> prj_dat = get_dgn_shp_prj(path_to_file=file_path)
        >>> type(prj_dat)
        dict
        >>> list(prj_dat.keys())
        ['PROJCS', 'Shapefile']
        >>> list(prj_dat['PROJCS'].keys())
        ['proj', 'lat_0', 'lon_0', 'k', 'x_0', 'y_0', 'ellps', 'units']
        >>> prj_dat['Shapefile'].shape
        (1, 4)
        >>> prj_dat = get_dgn_shp_prj(path_to_file=file_path, as_dict=False)
        >>> prj_dat['PROJCS'][:11]
        '+proj=tmerc'
    """

    path_to_file_ = path_to_file + (
        "_osr" if projcs_parser == 'osr' else "_pycrs") + ("_dict" if as_dict else "_proj4")
    path_to_pickle = path_to_file_ + ".pickle"

    if os.path.isfile(path_to_pickle) and not update:
        dgn_shp_prj = load_data(path_to_pickle)

    else:
        if verbose:
            print("Parsing projection metadata and boundary shapefile", end=" ... ")

        try:
            path_to_prj = path_to_file + ".prj"
            prj_file = parse_projcs(path_to_prj, parser=projcs_parser, as_dict=as_dict)

            path_to_shp = path_to_file + ".shp"
            shp_file = SHPReadParse.read_shp(path_to_shp, engine='gpd')

            dgn_shp_prj = {'PROJCS': prj_file, 'Shapefile': shp_file}

            if verbose:
                print("Done.")

            save_data(dgn_shp_prj, path_to_file=path_to_pickle, verbose=verbose)

        except Exception as e:
            print(f"Failed. {e}.")
            dgn_shp_prj = None

    return dgn_shp_prj


def dgn_shapefiles():
    """
    Get all types of shapefiles converted from a DGN file.

    :return: A list of shapefile types that can be converted from a DGN (.dgn) file.
    :rtype: list[str]

    **Examples**::

        >>> from src.utils import dgn_shapefiles
        >>> dgn_shapefiles()
        ['Annotation', 'MultiPatch', 'Point', 'Polygon', 'Polyline']
    """

    return ['Annotation', 'MultiPatch', 'Point', 'Polygon', 'Polyline']


def dgn_shapefile_ext():
    """
    Get all file extensions of shapefiles converted from a DGN file.

    :return: A list of file extensions for shapefiles converted from a DGN (.dgn) file.
    :rtype: list[str]

    **Examples**::

        >>> from src.utils import dgn_shapefile_ext
        >>> dgn_shapefile_ext()
        ['.cpg', '.dbf', '.sbn', '.sbx', '.shp', '.shp.xml', '.shx']
    """

    return ['.cpg', '.dbf', '.sbn', '.sbx', '.shp', '.shp.xml', '.shx']


def remove_dgn_shapefiles(path_to_dgn):
    """
    Remove all shapefiles converted from a .dgn file.

    :param path_to_dgn: The absolute path to the original .dgn data file.
    :type path_to_dgn: str

    .. seealso::

        :py:func:`~src.utils.dgn2shp_batch` for examples of this function in use.
    """

    dgn_filename = os.path.splitext(os.path.basename(path_to_dgn))[0].replace(".", "_")
    dat_dir = os.path.dirname(path_to_dgn)

    dat = [glob.glob(cd(dat_dir, "{}*{}".format(dgn_filename, ext))) for ext in dgn_shapefile_ext()]

    for dat_ in dat:
        for f in dat_:
            os.remove(f)


def dgn2shp(path_to_dgn, output_dir=None, export_to='shapefile', verbose=False, python2=None):
    """
    Convert a DGN file into shapefiles.

    The output includes a set of files with extensions such as .cpg, .dbf, .sbn, .sbx, .shp,
    .shp.xml, and .shx. The data is categorised into annotation, multipatch, point, polygon and
    polyline.

    This function uses the
    `arcpy <https://desktop.arcgis.com/en/arcmap/10.3/analyze/arcpy/what-is-arcpy-.htm>`_
    package available in ArcGIS Desktop.

    See also
    [`UD-1 <https://resources.arcgis.com/en/help/main/10.2/index.html#//001200000020000000>`_] and
    [`UD-2 <https://execnet.readthedocs.io/en/latest/example/hybridpython.html>`_].

    :param path_to_dgn: The path to the .dgn file.
    :type path_to_dgn: str
    :param output_dir: The absolute path to the folder where output files will be saved.
                       When ``output_dir=None`` (default), the files are saved in the same
                       directory as the .dgn file.
    :type output_dir: str | None
    :param export_to: The format to export the data to; defaults to ``'shapefile'``.
    :type export_to: str
    :param verbose: Whether to print relevant information to the console as the function runs;
                    defaults to ``False``.
    :type verbose: bool | int
    :param python2: The path to the Python 2.7 executable. When ``python2=None`` (default),
                    it defaults to ``"C:\\Python27\\ArcGIS10.7\\python"``.
    :type python2: str

    **Examples**::

        >>> from src.utils import dgn2shp
        >>> import glob
        >>> from pyhelpers.dirs import cd, cdd
        >>> temp_dat_dir = cdd("temp", "CARRS", "Tunnels")
        >>> opas_stn_dgn = cd(temp_dat_dir, "ENG_ADMIN.RINM_CARRS_TunnelPortal.dgn")
        >>> dgn2shp(opas_stn_dgn, verbose=True)
        Converting "ENG_ADMIN.RINM_CARRS_TunnelPortal.dgn" ... ... Done.
        >>> glob.glob1(temp_dat_dir, "*.shp")
        ['ENG_ADMIN_RINM_CARRS_TunnelPortal_dgn_Annotation.shp',
         'ENG_ADMIN_RINM_CARRS_TunnelPortal_dgn_MultiPatch.shp',
         'ENG_ADMIN_RINM_CARRS_TunnelPortal_dgn_Point.shp',
         'ENG_ADMIN_RINM_CARRS_TunnelPortal_dgn_Polygon.shp',
         'ENG_ADMIN_RINM_CARRS_TunnelPortal_dgn_Polyline.shp']
    """

    if python2 is None:
        python2 = "C:\\Python27\\ArcGIS10.7\\python"

    gw = execnet.makegateway("popen//python='%s'" % python2)

    if output_dir is None:
        output_dir = os.path.dirname(path_to_dgn).replace("\\", "/")

    if verbose:
        dgn_filename, dgn_rel_path = os.path.basename(path_to_dgn), os.path.relpath(output_dir)
        print("Converting \"%s\" at \"%s\"" % (dgn_filename, dgn_rel_path), end=" ... ")

    if export_to == 'shapefile':
        func = 'FeatureClassToShapefile_conversion'
    else:
        func = 'FeatureClassToGeodatabase_conversion'

    channel = gw.remote_exec(
        """
        import arcpy

        arcpy.%s("'%s/Annotation';'%s/MultiPatch';'%s/Point';'%s/Polygon';'%s/Polyline'", "%s")

        channel.send(None)
        """ % tuple([func] + [path_to_dgn.replace("\\", "/")] * 5 + [output_dir])
    )

    channel.send(None)

    channel.receive()

    channel.close()

    if verbose:
        print("Done.")


def dgn2shp_batch(dat_name, file_paths, confirmation_required=True, verbose=False, **kwargs):
    """
    Convert a list of DGN files to shapefiles.

    :param dat_name: Name of the data category.
    :type dat_name: str
    :param file_paths: A list of paths to DGN files.
    :type file_paths: list[str]
    :param confirmation_required: Whether to prompt for confirmation before proceeding;
                                  defaults to ``True``.
    :type confirmation_required: bool
    :param verbose: Whether to print relevant information to the console as the function runs;
                    defaults to ``False``.
    :type verbose: bool | int
    :param kwargs: [Optional] parameters of the :func:`~src.utils.dgn2shp` function.

    .. seealso::

        :py:meth:`~src.preprocessor.cnm.CNM.dgn2shp` for examples of the method.
    """

    if confirmed(f"To convert .dgn files of \"{dat_name}\" to shapefiles?\n",
                 confirmation_required=confirmation_required):
        for f in file_paths:
            shp_filenames = [f.replace(".", "_") + "_" + x + ".shp" for x in dgn_shapefiles()]

            if all(not os.path.isfile(f_) for f_ in shp_filenames):
                dgn2shp(f, verbose=verbose, **kwargs)

            else:
                dgn_filename = os.path.splitext(os.path.basename(f))[0]

                cfm_msg = f"Renew the existing .shp file for \"{dgn_filename}\"?"
                if confirmed(cfm_msg, confirmation_required=confirmation_required):
                    remove_dgn_shapefiles(f)

                    dgn2shp(f, verbose=verbose, **kwargs)


def get_field_names(path_to_file):
    """
    Get field names for a specific item.

    :param path_to_file: Path to the file that contains field names.
    :type path_to_file: str
    :return: A list of field names.
    :rtype: list

    **Examples**::

        >>> from src.utils import get_field_names
        >>> from pyhelpers.dirs import cdd
        >>> file_path = cdd("CARRS", "Overline bridges", "FID_Point.txt")
        >>> fn = get_field_names(file_path)
        >>> type(fn)
        list
        >>> fn[0:10]
        ['Entity',
         'Handle',
         'Level',
         'Layer',
         'LvlDesc',
         'LyrFrzn',
         'LyrLock',
         'LyrOn',
         'LvlPlot',
         'Color']
    """

    fn_file = pd.read_csv(path_to_file, sep='\t')

    field_names = fn_file['Field Name'].to_list()[2:]

    return field_names


def parse_dgn_shp_date(x):
    """
    Parse the data of dates in a shapefiles converted from a DGN data file.

    :param x: Date of a DGN shapefile.
    :type x: str | numpy.nan | None | float | int
    :return: Parsed date.
    :rtype: datetime.date

    **Examples**::

        >>> from src.utils import parse_dgn_shp_date
        >>> parse_dgn_shp_date('12/12/2010')
        datetime.date(2010, 12, 12)
        >>> parse_dgn_shp_date('12-DEC-10')
        datetime.date(2010, 12, 12)
    """

    try:
        if not pd.isna(x) and x != 'nan':
            try:
                date_dat = datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S.%f')
            except ValueError:
                date_dat = datetime.datetime.strptime(str(x), '%Y%m%d')
        else:
            date_dat = pd.NaT

    except ValueError:
        try:
            date_dat = dateutil.parser.parse(x, dayfirst=True).date()
        except (dateutil.parser.ParserError, TypeError):
            date_dat = pd.NaT

    return date_dat


def read_dgn_shapefile(path_to_dir, dgn_filename):
    """
    Read the shapefiles converted from a DGN data file.

    :param path_to_dir: Path to the data directory.
    :type path_to_dir: str
    :param dgn_filename: Filename of the original DGN data.
    :type dgn_filename: str
    :return: Data of the converted shapefiles.
    :rtype: dict

    .. seealso::

        Examples for the following methods of the class :class:`~src.preprocessor.CARRS`:
          - :py:meth:`~src.preprocessor.CARRS.read_overline_bridges_shp`
          - :py:meth:`~src.preprocessor.CARRS.read_underline_bridges_shp`
          - :py:meth:`~src.preprocessor.CARRS.read_retaining_walls_shp`
          - :py:meth:`~src.preprocessor.CARRS.read_tunnels_shp`
    """

    shp_ext = ".shp"

    shp_filenames = [dgn_filename.replace(".", "_") + "_" + x + shp_ext for x in dgn_shapefiles()]
    shp_data = [SHPReadParse.read_shp(cd(path_to_dir, sfn), engine='gpd') for sfn in shp_filenames]

    field_names_filenames = [x.replace(shp_ext, ".txt") for x in shp_filenames]
    field_names = [
        get_field_names(cd(path_to_dir, "FID_" + ffn.split("_")[-1])) for ffn in field_names_filenames]

    for shp_dat, field_name in zip(shp_data, field_names):
        if len(shp_dat.columns) > len(field_name):
            field_name.append('geometry')
        shp_dat.columns = field_name

        columns_to_drop = [
            'Layer', 'LvlDesc', 'Class', 'CadModel', 'DocName', 'DocPath', 'DocType', 'DocVer']
        shp_dat.drop(columns=columns_to_drop, inplace=True)

        # Parse date
        if not shp_dat.empty:
            date_columns = [x for x in field_name if x.endswith('DATE')]

            # cols_with_na = shp_dat[date_columns].columns[
            #     shp_dat[date_columns].isna().any()].tolist()
            # cols_without_na = [x for x in date_columns if x not in cols_with_na]

            # def parse_dgn_shp_date_v2(x):
            #     try:
            #         date_dat = dateutil.parser.parse(x, dayfirst=True)
            #     except (dateutil.parser.ParserError, TypeError):
            #         date_dat = pd.NaT
            #     return date_dat

            # try:
            dates_dat = shp_dat[date_columns].map(parse_dgn_shp_date)
            # except ValueError:
            #     dates_dat = shp_dat[date_columns].applymap(parse_dgn_shp_date_v2)

            shp_dat[date_columns] = dates_dat

    dgn_shp_data = dict(zip(dgn_shapefiles(), shp_data))

    return dgn_shp_data


def dgn_shp_map_view(cls_instance, item_name, layer_name, desc_col_name, sample=True,
                     marker_colour='blue', update=False, verbose=True):
    """
    Get a map view of a specific item.

    :param cls_instance: Instance of a data class.
    :type cls_instance: src.preprocessor.CARRS | src.preprocessor.CNM | src.preprocessor.OPAS
    :param item_name: Name of an item.
    :type item_name: str
    :param layer_name: Name of a layer.
    :type layer_name: str
    :param desc_col_name: Name of a column that describes markers.
    :type desc_col_name: str
    :param sample: Whether to draw a sample, or a specific sample size.
    :type sample: bool | int
    :param marker_colour: Colour of markers; defaults to ``'blue'``.
    :type marker_colour: str
    :param update: Whether to read the original data
        (instead of loading its pickle file if available); defaults to ``False``.
    :type update: bool
    :param verbose: Whether to print relevant information in console as the function runs;
        defaults to ``True``.
    :type verbose: bool | int

    .. seealso::

        Examples for the following methods:
          - :meth:`CARRS.map_view()<src.preprocessor.CARRS.map_view>`
          - :meth:`CNM.map_view()<src.preprocessor.CNM.map_view>`
          - :meth:`OPAS.map_view()<src.preprocessor.OPAS.map_view>`
    """

    item_names = [getattr(cls_instance, x) for x in dir(cls_instance) if x.endswith('TblName')]
    if item_name is None:
        item_name_ = input("Choose one of {}?".format(set(item_names)))
    else:
        item_name_ = find_similar_str(item_name, item_names)
    item_name_ = item_name_.replace(' ', '_').lower()

    m_filename = "view_{}.html".format(item_name_)
    folder_names = [getattr(cls_instance, x) for x in dir(cls_instance) if x.endswith('DirName')]
    path_to_m = cd(cls_instance.DATA_DIR, find_similar_str(item_name, folder_names), m_filename)

    if not os.path.isfile(path_to_m) or update:

        shp_data = getattr(cls_instance, f'read_{item_name_}_shp')(update=update, verbose=verbose)
        shp_dat = shp_data[layer_name]

        if sample:
            n_sample = 100 if type(sample) is bool else sample
            shp_dat = shp_dat.sample(n=n_sample, random_state=1)

        lonlat = np.array(osgb36_to_wgs84(shp_dat.geometry.x, shp_dat.geometry.y)).T

        dgn_shp_prj = cls_instance.read_prj_metadata()
        proj_shp = dgn_shp_prj['Shapefile']

        min_x, min_y, max_x, max_y = proj_shp.total_bounds

        ll_lon, ll_lat = osgb36_to_wgs84(min_x, min_y)
        ur_lon, ur_lat = osgb36_to_wgs84(max_x, max_y)

        m = folium.Map(
            location=[(ll_lat + ur_lat) / 2, (ur_lon + ll_lon) / 2], zoom_start=6,
            control_scale=True)

        for coord, description in zip(lonlat, shp_dat[desc_col_name]):
            folium.Marker(
                location=list(reversed(coord)),
                popup=folium.Popup(f'{desc_col_name}: {description}', max_width=500),
                icon=folium.Icon(color=marker_colour),
            ).add_to(m)

        folium.plugins.MiniMap(zoom_level_offset=-6).add_to(m)

        folium.plugins.FloatImage(
            image=cdd("_backups\\Misc", "north-arrow.png"),  # bottom=1, left=8,  # width='1.5'
            bottom=8, left=1, width='22px').add_to(m)

        m.save(path_to_m)

        fix_folium_float_image(path_to_m)

        shutil.copyfile(path_to_m, cd("docs\\source\\_static", m_filename))

    webbrowser.open(path_to_m)
