"""
This module provides helper classes/functions for dealing with LAZ/LAS data.
"""

import os
import subprocess
import tempfile
import urllib.parse

from pyhelpers.dirs import cd, validate_dir
from pyhelpers.ops import download_file_from_url
from pyhelpers.text import find_similar_str


def download_las_tools(las_tool=None, download_dir=None, sub_dir="", update=False, verbose=True,
                       ret_tool_path=False, **kwargs):
    """
    Download a LAS tool from https://lastools.github.io/.

    :param las_tool: Name of the LAS tool to be downloaded;
        options include ``{'LAStools.zip', 'laszip.exe', 'laszip64.exe', 'laszip-cli.exe'}``;
        when ``las_tool=None`` (default), it is to use/download "laszip64.exe".
    :type las_tool: str | None
    :param download_dir: Name of a directory for the downloaded LAS tool; defaults to ``None``.
    :type download_dir: str | None
    :param sub_dir: Name of a subdirectory; defaults to ``""``.
    :type sub_dir: str
    :param update: Whether to update the existing data file(s); defaults to ``False``.
    :type update: bool
    :param verbose: Whether to print relevant information in console as the function runs;
        defaults to ``True``.
    :type verbose: bool | int
    :param ret_tool_path: Whether to return the absolute path to the downloaded file;
        defaults to ``False``.
    :type ret_tool_path: bool
    :param kwargs: [Optional] parameters of `pyhelpers.ops.download_file_from_url
        <https://pyhelpers.readthedocs.io/en/latest/_generated/
        pyhelpers.ops.download_file_from_url.html#pyhelpers.ops.download_file_from_url>`_.
    :return: An absolute path to the downloaded file (only when ``ret_tool_path=True``).
    :rtype: str

    **Examples**::

        >>> from src.utils import download_las_tools
        >>> import os
        >>> from pyhelpers.dirs import cdd
        >>> dwnld_dir = cdd("PCD")
        >>> # # download_las_tools('laszip', dwnld_dir, update=True, verbose=True)
        >>> download_las_tools('laszip', dwnld_dir)
        Downloading "laszip.exe" ... Done.
        >>> # "laszip.exe" already exists at "data\\PCD"
        >>> path_to_laszip64 = download_las_tools('laszip64', dwnld_dir, ret_tool_path=True)
        "laszip64.exe" already exists at "data\\PCD"
        >>> os.path.relpath(path_to_laszip64)
        'data\\PCD\\laszip64.exe'
    """

    if las_tool is None:
        las_tool_ = 'laszip64.exe'
    else:
        tools = ['LAStools.zip', 'laszip.exe', 'laszip64.exe', 'laszip-cli.exe']
        las_tool_ = find_similar_str(las_tool, tools)

    download_page = 'https://lastools.github.io/download/'
    download_url = urllib.parse.urljoin(download_page, las_tool_)

    download_dir_ = validate_dir(download_dir, sub_dir=sub_dir)
    path_to_las_tool = cd(download_dir_, las_tool_)

    if os.path.exists(path_to_las_tool) and not update:
        if verbose:
            print("\"{}\" already exists at \"{}\"".format(
                las_tool_, os.path.dirname(os.path.relpath(path_to_las_tool))))

    else:
        if verbose:
            print("Downloading \"{}\"".format(las_tool_), end=" ... ")

        try:
            download_file_from_url(download_url, path_to_las_tool, verbose=False, **kwargs)

            if os.path.exists(path_to_las_tool) and verbose:
                print("Done.")

        except Exception as e:
            if verbose:
                print(f"Failed. {e}.")

    if ret_tool_path:
        return path_to_las_tool


def laz2las(laz_files, output_las=None, temp_file_list=False, laszip_exe="laszip64.exe"):
    """
    Decompress a LAZ file to a LAS file.

    :param laz_files: Path(s) to source LAZ file(s).
    :type laz_files: str
    :param output_las: Path to a LAS file as the output; defaults to ``None``.
    :type output_las: str
    :param temp_file_list: Whether to use a temporary .txt file to list all LAZ files;
        defaults to ``False``.
    :type temp_file_list: bool
    :param laszip_exe: Path to the executable file `laszip64`_; defaults to ``"laszip64.exe"``.
    :type laszip_exe: str

    See also [`L2L-1`_] and [`L2L-2`_].

    .. _`laszip64`: https://lastools.github.io/download/laszip64.exe
    .. _`L2L-1`: https://rapidlasso.com/laszip/
    .. _`L2L-2`: https://www.cs.unc.edu/~isenburg/lastools/download/laszip_README.txt

    **Examples**::

        >>> from src.utils import laz2las
        >>> import os
        >>> from pyhelpers.dirs import cd, cdd
        >>> pcd_dir = cdd("PCD")
        >>> laz_dir = cd(pcd_dir, "ECM8\\LAZ_OSGB_100x100\\201910")
        >>> filename = "Tile_X+0000340100_Y+0000674000.laz"
        >>> laszip_exe_ = cd(pcd_dir, "laszip.exe")

        >>> # -- Example 1 -----------------------------------------------------------------------
        >>> laz_file = cd(laz_dir, filename)  # path to the source LAZ file
        >>> las_file = laz_file.replace(".laz", ".las")  # path to the output LAS file
        >>> # Decompress the (source) LAZ file to the (output) LAS file
        >>> laz2las(laz_file, las_file, laszip_exe=laszip_exe_)

        >>> # -- Example 2 -----------------------------------------------------------------------
        >>> laz_wild_card = os.path.relpath(cd(laz_dir, "*.laz"))  # paths to the source LAZ files
        >>> laz2las(laz_wild_card, laszip_exe=laszip_exe_)
    """

    if isinstance(laz_files, list):
        if temp_file_list:
            file_list = tempfile.NamedTemporaryFile(suffix=".txt", mode='w', delete=False)
            file_list.writelines('\n'.join(laz_files))
            file_list.close()
            laz_files_ = '"{}"'.format(file_list.name)
            # laz_files_ = cd(os.path.commonpath(laz_files), "file_list.txt")
            # file_list = open(laz_files_, "w")
            # file_list.writelines('\n'.join(laz_files))
            # file_list.close()
        else:
            laz_files_ = '"' + '" "'.join(laz_files) + '"'
    else:
        assert laz_files.endswith(".laz") or laz_files.endswith(".txt")
        laz_files_ = '"{}"'.format(os.path.relpath(laz_files))

    if laz_files_.endswith('.txt"'):
        in_ = '-lof'
    else:
        in_ = '-i'

    if output_las:
        output_las_ = '-o "{}"'.format(output_las)
    elif laz_files_.endswith('*.laz"') or laz_files_.endswith('.txt"') \
            or '" "' in laz_files_:
        output_las_ = ''
    else:
        output_ = os.path.basename(laz_files_).replace('.laz', '.las')
        output_las_ = '-o "{}"'.format(output_)

    cmd = '"{}" {} {} {}'.format(laszip_exe, in_, laz_files_, output_las_).strip()
    subprocess.call(cmd)

    if temp_file_list and laz_files_.endswith(".txt"):
        os.remove(laz_files_)


def merge_las_files(las_files, output_las=None, laszip_exe="laszip64.exe", ret_output_path=False):
    """
    Merge multiple LAS files into a single LAS file.

    :param las_files: Paths to LAS files or a path to a .txt file that contains a list of LAS files.
    :type las_files: list | str
    :param output_las: Path to a LAS file; defaults to ``None``.
    :type output_las: str | None
    :param laszip_exe: Path to the `laszip`_ executable; defaults to ``"laszip64.exe"``.
    :type laszip_exe: str
    :param ret_output_path: Whether to return the path to the output merged LAS data file
    :type ret_output_path: bool
    :return: (Only if ``ret_output_path=True``) an absolute path to the output merged LAS data file.
    :rtype: str

    .. _`laszip`: https://lastools.github.io/download/laszip.exe

    See also [`L2L-1`_] and [`L2L-2`_].

    .. _`L2L-1`: https://rapidlasso.com/laszip/
    .. _`L2L-2`: https://www.cs.unc.edu/~isenburg/lastools/download/laszip_README.txt

    **Examples**::

        >>> from src.utils import laz2las, merge_las_files
        >>> import os
        >>> from pyhelpers.dirs import cd, cdd
        >>> laszip_exe_ = cdd("PCD\\laszip.exe")
        >>> pcd_dir = cdd("PCD")
        >>> laz_dir = cd(pcd_dir, "ECM8\\LAZ_OSGB_100x100")
        >>> # Decompress LAZ files to LAS files
        >>> laz_wild_card = os.path.relpath(cd(laz_dir, "*.laz"))
        >>> laz2las(laz_wild_card, laszip_exe=laszip_exe_)
        >>> # Merge the LAS files into a LAS file
        >>> las_wild_card = os.path.relpath(cd(laz_dir, "*.las"))
        >>> merged_file = merge_las_files(
        ...     las_wild_card, laszip_exe=laszip_exe_, ret_output_path=True)
        >>> os.path.relpath(merged_file, pcd_dir)
        'ECM8\\LAZ_OSGB_100x100\\LAZ_OSGB_100x100.las'
    """

    if isinstance(las_files, list):
        las_files_ = '"' + '" "'.join(las_files) + '"'
    else:
        las_files_ = '"{}"'.format(las_files)

    if output_las:
        output_las_ = '"{}"'.format(output_las)
    else:
        if isinstance(las_files, list):
            output_path = os.path.commonpath(las_files)
        else:
            output_path = os.path.dirname(las_files)
        output_las_ = '"{}\\{}.las"'.format(output_path, os.path.basename(output_path))

    cmd = '"{}" -i {} -merged -o {}'.format(laszip_exe, las_files_, output_las_).strip()

    subprocess.call(cmd)

    if ret_output_path:
        return os.path.abspath(output_las_.strip('"'))
