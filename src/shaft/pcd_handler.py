"""
This module processes the point cloud data handled by the class :class:`~src.preprocessor.pcd.PCD`.
"""

import datetime

import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import shapely.ops
import shapely.wkt
from pyhelpers._cache import _print_failure_msg
from pyhelpers.dirs import cd
from pyhelpers.store import _check_saving_path, save_figure
from shapely.geometry import Polygon

from src.preprocessor import PCD
from src.utils.general import cd_docs_source, get_tile_xy


class PCDHandler(PCD):
    """
    Explore and process the point cloud data.
    """

    #: Descriptive name of the class.
    NAME: str = 'Handler of preprocessed point cloud data'

    def __init__(self, db_instance=None):
        """
        :param db_instance: PostgreSQL database instance; defaults to ``None``.
        :type db_instance: src.utils.TrackFixityDB | None

        **Examples**::

            >>> from src.shaft import PCDHandler
            >>> pcdh = PCDHandler()
            >>> pcdh.NAME
            'Handler of preprocessed point cloud data'
        """

        super().__init__(db_instance=db_instance)

    @staticmethod
    def o3d_transform(xyz_rgb_data, greyscale=False, voxel_size=None):
        """
        Transform the numpy array of point cloud (XYZ and RGB) data to Open3D object(s)
        (open3d.geometry.PointCloud).

        :param xyz_rgb_data: XYZ and RGB data.
        :type xyz_rgb_data: dict
        :param voxel_size: Voxel size for downsampling; defaults to ``None``.
        :type voxel_size: float | None
        :param greyscale: Whether to transform the colour data to greyscale; defaults to ``False``.
        :type greyscale: bool
        :return: Open3D objects of point cloud data.
        :rtype: tuple

        **Examples**::

            >>> from src.shaft import PCDHandler
            >>> from pyhelpers.settings import np_preferences
            >>> np_preferences()
            >>> pcdh = PCDHandler()
            >>> tile_xy = (340500, 674200)
            >>> rows_limit = 1000
            >>> # Data of '202004'
            >>> xyz_rgb_dat = pcdh.load_laz(tile_xy, pcd_dates=['202004'], limit=rows_limit)
            >>> type(xyz_rgb_dat)
            dict
            >>> list(xyz_rgb_dat.keys())
            ['202004']
            >>> len(xyz_rgb_dat['202004'])
            2
            >>> type(xyz_rgb_dat['202004'][0])
            numpy.ndarray
            >>> xyz_rgb_dat['202004'][0].shape
            (1000, 3)
            >>> type(xyz_rgb_dat['202004'][1])
            numpy.ndarray
            >>> xyz_rgb_dat['202004'][1].shape
            (1000, 3)
            >>> # Original data of all dates available
            >>> xyz_rgb_dat = pcdh.load_laz(tile_xy, limit=rows_limit)
            >>> list(xyz_rgb_dat.keys())
            ['201910', '202004']
            >>> # Transform the array data to open3d.geometry.PointCloud class
            >>> o3d_pcd_lst = pcdh.o3d_transform(xyz_rgb_dat)
            >>> o3d_pcd_lst
            [PointCloud with 1000 points., PointCloud with 1000 points.]
            >>> # Downsample the data with voxels
            >>> o3d_pcd_ds_lst = pcdh.o3d_transform(xyz_rgb_dat, voxel_size=0.05)
            >>> o3d_pcd_ds_lst
            [PointCloud with 915 points., PointCloud with 122 points.]
        """

        o3d_pcd_list = []

        for _, xyz_rgb in xyz_rgb_data.items():
            xyz, rgb = xyz_rgb

            # Open3D object
            o3d_pcd = o3d.geometry.PointCloud()

            # Transform the data from Numpy to Open3D (o3d.geometry.PointCloud) type
            o3d_pcd.points = o3d.utility.Vector3dVector(xyz)

            if greyscale:
                o3d_pcd.colors = o3d.utility.Vector3dVector(rgb / 255)

            if voxel_size is not None:
                o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)

            # if paint_colour is not None and pcd_date == '202004' and len(xyz_rgb_data) > 1:
            #     o3d_pcd.paint_uniform_color(paint_colour)

            o3d_pcd_list.append(o3d_pcd)

        return o3d_pcd_list

    def view_pcd_example(self, tile_xy, pcd_dates=None, limit=None, voxel_size=None,
                         greyscale=False, gs_coef=1.2, save_as=None, verbose=False, ret_sample=True,
                         **kwargs):
        """
        Visualise a sample (tile) of point cloud data.

        :param tile_xy: X and Y coordinates in reference to a tile for the point cloud data.
        :type tile_xy: tuple | list | str
        :param pcd_dates: Date(s) of the point cloud data; defaults to ``None``.
        :type pcd_dates: str | list | None
        :param voxel_size: Voxel size for downsampling; defaults to ``0.05``.
        :type voxel_size: float | None
        :param greyscale: Whether to transform the colour data to greyscale; defaults to ``False``.
        :type greyscale: bool
        :param gs_coef: Coefficient associated with intensity (only if ``greyscale=True``),
            defaults to ``1.2``.
        :type gs_coef: float
        :param limit: Limit on the number of rows to query from the database; defaults to ``None``.
        :type limit: int
        :param save_as: File format that the view is saved as; defaults to ``None``.
        :type save_as: str | None
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param ret_sample: Whether to return the sample data.
        :type ret_sample: bool
        :param kwargs: [Optional] parameters for `open3d.visualization.draw_geometries`_.
        :return: The Open3D object(s) of the sample point cloud data (only if ``ret_sample=True``).
        :rtype: tuple

        .. _`open3d.visualization.draw_geometries`:
            http://www.open3d.org/docs/release/python_api/open3d.visualization.draw_geometries.html

        **Examples**::

            >>> from src.shaft import PCDHandler
            >>> pcdh = PCDHandler()
            >>> xy_tile = (340500, 674200)
            >>> pcd1 = pcdh.view_pcd_example(xy_tile, '201910', greyscale=True, width=1000)

        .. figure:: ../_images/pcdh_view_pcd_x340500y674200_gs_201910.*
            :name: pcdh_view_pcd_x340500y674200_gs_201910
            :align: center
            :width: 100%

            Point cloud data (greyscale, October 2019) of tile (340500, 674200).

        .. code-block:: python

            >>> pcd1
            [PointCloud with 54007138 points.]

        .. code-block:: python

            >>> pcd2 = pcdh.view_pcd_example(xy_tile, pcd_dates='201910', width=1000)

        .. figure:: ../_images/pcdh_view_pcd_x340500y674200_201910.*
            :name: pcdh_view_pcd_x340500y674200_201910
            :align: center
            :width: 100%

            Point cloud data (coloured, October 2019) of tile (340500, 674200).

        .. code-block:: python

            >>> pcd2
            [PointCloud with 54007138 points.]

        .. code-block:: python

            >>> pcd3 = pcdh.view_pcd_example(xy_tile, '202004', greyscale=True, width=1000)

        .. figure:: ../_images/pcdh_view_pcd_x340500y674200_gs_202004.*
            :name: pcdh_view_pcd_x340500y674200_gs_202004
            :align: center
            :width: 100%

            Point cloud data (greyscale, April 2020) of tile (340500, 674200).

        .. code-block:: python

            >>> pcd3
            [PointCloud with 20505100 points.]

        .. code-block:: python

            >>> pcd4 = pcdh.view_pcd_example(xy_tile, pcd_dates='202004', width=1000)

        .. figure:: ../_images/pcdh_view_pcd_x340500y674200_202004.*
            :name: pcdh_view_pcd_x340500y674200_202004
            :align: center
            :width: 100%

            Point cloud data (coloured, April 2020) of tile (340500, 674200).

        .. code-block:: python

            >>> pcd4
            [PointCloud with 20505100 points.]
        """

        xyz_rgb_data = self.load_laz(
            tile_xy=tile_xy, pcd_dates=pcd_dates, greyscale=greyscale, gs_coef=gs_coef, limit=limit)

        o3d_pcd_list = self.o3d_transform(
            xyz_rgb_data=xyz_rgb_data, greyscale=greyscale, voxel_size=voxel_size)

        tile_x, tile_y = get_tile_xy(tile_xy)

        window_name = f'Tile_X+0000{tile_x}_Y+0000{tile_y}'
        if voxel_size is not None and len(o3d_pcd_list) >= 1:
            window_name += ' (voxel_down_sample)'

        # o3d.visualization.draw_geometries(o3d_pcd_list, window_name=window_name, **kwargs)

        vis = o3d.visualization.Visualizer()

        # def rotate_view(vis):
        #     ctr = vis.get_view_control()
        #     ctr.rotate(10.0, 0.0)
        #     return False
        #
        # o3d.visualization.draw_geometries_with_animation_callback(
        #     o3d_pcd_list, rotate_view, window_name=window_name)

        vis.create_window(window_name=window_name, **kwargs)  # optional: width=1200, height=860

        for pcd_geom in o3d_pcd_list:
            vis.add_geometry(pcd_geom)
            vis.update_geometry(pcd_geom)

        vis.poll_events()
        vis.update_renderer()

        # img_depth = vis.capture_depth_float_buffer(False)
        # img_image = vis.capture_screen_float_buffer(False)

        if save_as:
            img_filename = f"tm_view_pcd_x{tile_x}y{tile_y}" + ("_gs" if greyscale else "")
            if pcd_dates:
                img_fn_suffix = pcd_dates if isinstance(pcd_dates, str) else "_".join(pcd_dates)
                img_filename += f"_{img_fn_suffix}{save_as}"
            path_to_img = cd(f"docs\\source\\_images\\{img_filename}")

            _check_saving_path(path_to_img, verbose=verbose, ret_info=False)

            try:
                # plt.imsave(path_to_img, np.asarray(img_depth))
                # plt.imsave(path_to_img, np.asarray(img_image))
                vis.capture_screen_image(path_to_img)
                if verbose:
                    print("Done.")
            except Exception as e:
                _print_failure_msg(e, msg="Failed.")

        vis.run()  # This allows the image window to remain

        vis.destroy_window()

        if ret_sample:
            return o3d_pcd_list

    def get_pcd_tile_polygon(self, tile_xy, pcd_date=None, as_geom=True):
        """
        Make a polygon for a given (X, Y) in reference to a tile.

        :param tile_xy: X and Y coordinates in reference to a tile for the point cloud data.
        :type tile_xy: tuple | list | str
        :param pcd_date: Date of the point cloud data; defaults to ``None``.
        :type pcd_date: str | int | None
        :param as_geom: Whether to return the polygon as a geometry object; defaults to ``True``.
        :type as_geom: bool
        :return: Polygon for the given (X, Y) in reference to a tile.
        :rtype: shapely.geometry.Polygon | numpy.ndarray

        **Examples**::

            >>> from src.shaft import PCDHandler
            >>> pcdh = PCDHandler()
            >>> tile_x_y = (340500, 674200)
            >>> tile_polygon = pcdh.get_pcd_tile_polygon(tile_xy=tile_x_y)
            >>> type(tile_polygon)
            shapely.geometry.polygon.Polygon
            >>> print(tile_polygon.wkt)
            POLYGON ((340500 674200, 340500 674300, 340600 674300, 340600 674200, 340500 674200))
            >>> tile_polygon = pcdh.get_pcd_tile_polygon(tile_xy=tile_x_y, as_geom=False)
            >>> type(tile_polygon)
            numpy.ndarray
            >>> tile_polygon
            array([[340500, 674200],
                   [340500, 674300],
                   [340600, 674300],
                   [340600, 674200],
                   [340500, 674200]])

        **Illustration**::

            import matplotlib.pyplot as plt
            from pyhelpers.settings import mpl_preferences

            mpl_preferences(backend='TkAgg', font_name='Times New Roman')

            fig = plt.figure(constrained_layout=True)
            ax = fig.add_subplot()
            ax.xaxis.set_ticks([tile_x_y[0], tile_x_y[0] + 100])
            ax.yaxis.set_ticks([tile_x_y[1], tile_x_y[1] + 100])

            c1, c2 = plt.get_cmap('tab10').colors[:2]

            ax.plot(tile_polygon[:, 0], tile_polygon[:, 1], linewidth=3)
            tile_label = f'Tile for {tile_x_y}'
            ax.scatter([], [], marker='s', s=200, fc='none', ec=c1, lw=3, label=tile_label)
            ax.scatter(tile_x_y[0], tile_x_y[1], s=100, color=c2, label=f'{tile_x_y}')

            ax.legend(loc='best')

            fig.show()

            # from pyhelpers.store import save_figure
            # fig_pathname = "docs/source/_images/pcdh_get_pcd_tile_polygon_demo"
            # save_figure(fig, f"{fig_pathname}.svg", verbose=True)
            # save_figure(fig, f"{fig_pathname}.pdf", verbose=True)

        .. figure:: ../_images/pcdh_get_pcd_tile_polygon_demo.*
            :name: pcdh_get_pcd_tile_polygon_demo
            :align: center
            :width: 80%

            Tile of (340100, 674000).
        """

        tile_x, tile_y = get_tile_xy(tile_xy=tile_xy)

        try:
            pcd_tiles = self.load_tiles(pcd_date=pcd_date)

            tile_name = f'Tile_X+0000{tile_x}_Y+0000{tile_y}'
            tile_poly = pcd_tiles[pcd_tiles.Tile_Name == tile_name].Tile_XY.drop_duplicates()

            tile_poly = shapely.wkt.loads(tile_poly.values[0])

            if not as_geom:
                tile_poly = np.array(tile_poly.exterior.coords).astype(int)

        except IndexError:
            tile_poly = np.array([(tile_x, tile_y),
                                  (tile_x, tile_y + 100),
                                  (tile_x + 100, tile_y + 100),
                                  (tile_x + 100, tile_y),
                                  (tile_x, tile_y)])

            if as_geom:
                tile_poly = Polygon(tile_poly)

        return tile_poly

    def get_pcd_tile_mpl_path(self, tile_xy, pcd_date=None, **kwargs):
        """
        Make a `matplotlib.path.Path <https://matplotlib.org/stable/api/path_api.html>`_ object
        for a tile for the point cloud data.

        :param tile_xy: X and Y coordinates in reference to a tile for the point cloud data.
        :type tile_xy: tuple | list | str
        :param pcd_date: Date of the point cloud data; defaults to ``None``.
        :type pcd_date: str | int | None
        :param kwargs: [Optional] parameter used by `matplotlib.path.Path`_.
        :return: `matplotlib.path.Path`_ object for the tile labelled ``tile_xy``.
        :rtype: matplotlib.path.Path

        .. _`matplotlib.path.Path`: https://matplotlib.org/stable/api/path_api.html

        **Examples**::

            >>> from src.shaft import PCDHandler
            >>> pcdh = PCDHandler()
            >>> # Make a polygon (i.e. a matplotlib.Path instance) of an example tile
            >>> tile_frame = pcdh.get_pcd_tile_mpl_path(tile_xy=(340100, 674000), pcd_date='202004')
            >>> type(tile_frame)
            matplotlib.path.Path
            >>> tile_frame
            Path(array([[340100.0000, 674000.0000],
                   [340100.0000, 674100.0000],
                   [340200.0000, 674100.0000],
                   [340200.0000, 674000.0000],
                   [340100.0000, 674000.0000]]), array([ 1,  2,  2,  2, 79], dtype=uint8))
        """

        tile_vertices = self.get_pcd_tile_polygon(tile_xy=tile_xy, pcd_date=pcd_date, as_geom=False)

        tile_polyline_path = matplotlib.path.Path(
            vertices=tile_vertices.reshape(5, 2), closed=True, **kwargs)

        return tile_polyline_path

    @staticmethod
    def _save_pcd_dgn_shp_view(fig, save_as, fig_filename_, dpi, verbose):
        """
        Save the plot of point cloud data from DGN-converted shapefile.

        :param save_as: File format that the view is saved as; defaults to ``None``.
        :type save_as: str | None
        :param fig_filename_: Filename (without the file extension).
        :type fig_filename_: str
        :param dpi: DPI for saving image; defaults to ``600``.
        :type dpi: int | None
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        """

        if save_as:
            for save_as_ in {save_as, ".svg", ".pdf"}:
                path_to_fig = cd_docs_source("_images", fig_filename_ + save_as_)
                save_figure(fig, path_to_fig, dpi=dpi, transparent=True, verbose=verbose)

    def _view_pcd_dgn(self, dgn_ls, dgn_cc, projection, colours, add_title, labels,
                      tile_name, title_date, save_as, fig_filename_, dpi, verbose, **kwargs):
        if projection:
            fig1 = plt.figure(figsize=(8, 7))
            ax1 = fig1.add_subplot(projection=projection)
            ax1.tick_params(axis='x', which='major', pad=8)
            ax1.tick_params(axis='y', which='major', pad=8)
            ax1.tick_params(axis='z', which='major', pad=10)
            ax1.scatter(
                dgn_cc[:, 0], dgn_cc[:, 1], dgn_cc[:, 2], marker='x', color=colours[0],
                label=labels[0], **kwargs)
            ax1.scatter(
                dgn_ls[:, 0], dgn_ls[:, 1], dgn_ls[:, 2], marker='^', color=colours[1],
                label=labels[1], **kwargs)
            ax1.set_xlabel('Easting', fontsize=13, labelpad=15)
            ax1.set_ylabel('Northing', fontsize=13, labelpad=15)
            ax1.set_zlabel('Elevation', fontsize=13, labelpad=12)
            plt.subplots_adjust(left=0, bottom=0, right=0.92, top=1)

        else:
            fig1 = plt.figure(figsize=(11, 5))
            ax1 = fig1.add_subplot(aspect='equal', adjustable='box')
            ax1.scatter(
                dgn_cc[:, 0], dgn_cc[:, 1], marker='x', color=colours[0], label=labels[0], **kwargs)
            ax1.scatter(
                dgn_ls[:, 0], dgn_ls[:, 1], marker='^', color=colours[1], label=labels[1], **kwargs)
            ax1.set_xlabel('Easting', fontsize=13, labelpad=8)
            ax1.set_ylabel('Northing', fontsize=13, labelpad=8)
            ax1.grid()

        ax1.ticklabel_format(useOffset=False)

        if add_title:
            ax1.set_title(f"DGN-PointCloud data of Tile{tile_name} in {title_date}.")

        ax1.legend(loc='best', numpoints=1, ncol=1, fontsize=11)

        fig1.tight_layout()

        self._save_pcd_dgn_shp_view(
            fig1, save_as=save_as, fig_filename_=fig_filename_, dpi=dpi, verbose=verbose)

    def _view_pcd_dgn_common_points(self, dgn_ls, dgn_cc, projection, colours, add_title, labels,
                                    tile_name, title_date, save_as, fig_filename_, dpi, verbose,
                                    **kwargs):
        """Common points between ``'Complex Chain'`` and ``'LineString'`` entities."""
        common_points = np.array(
            [p for p in set([tuple(p) for p in dgn_ls]) & set([tuple(p) for p in dgn_cc])])

        if projection:
            fig2 = plt.figure(figsize=(8, 7))
            ax2 = fig2.add_subplot(projection=projection)
            ax2.tick_params(axis='x', which='major', pad=8)
            ax2.tick_params(axis='y', which='major', pad=8)
            ax2.tick_params(axis='z', which='major', pad=10)
            ax2.scatter(
                common_points[:, 0], common_points[:, 1], common_points[:, 2], color=colours[3],
                **kwargs)
            ax2.set_xlabel('Easting', fontsize=13, labelpad=15)
            ax2.set_ylabel('Northing', fontsize=13, labelpad=15)
            ax2.set_zlabel('Elevation', fontsize=13, labelpad=12)
            plt.subplots_adjust(left=0, bottom=0, right=0.92, top=1)

        else:
            fig2 = plt.figure(figsize=(11, 5))
            ax2 = fig2.add_subplot(aspect='equal', adjustable='box')
            ax2.scatter(common_points[:, 0], common_points[:, 1], color=colours[3], **kwargs)
            ax2.set_xlabel('Easting', fontsize=13, labelpad=8)
            ax2.set_ylabel('Northing', fontsize=13, labelpad=8)
            ax2.grid()

        ax2.ticklabel_format(useOffset=False)
        if add_title:
            ax2.set_title(
                f'Common points between "{labels[0]}" and "{labels[1]}" entities of polyline\n'
                f'of Tile{tile_name} in {title_date}.')

        fig2.tight_layout()

        self._save_pcd_dgn_shp_view(
            fig2, save_as=save_as, fig_filename_=fig_filename_, dpi=dpi, verbose=verbose)

    def _view_pcd_dgn_unique_points(self, dgn_ls, dgn_cc, projection, colours, add_title, labels,
                                    tile_name, title_date, save_as, fig_filename_, dpi, verbose,
                                    **kwargs):
        """Unique points between ``'Complex Chain'`` and ``'LineString'`` entities."""

        unique_points = np.unique(np.vstack([dgn_ls, dgn_cc]), axis=0)

        if projection:
            fig3 = plt.figure(figsize=(8, 7))
            ax3 = fig3.add_subplot(projection=projection)
            ax3.tick_params(axis='x', which='major', pad=8)
            ax3.tick_params(axis='y', which='major', pad=8)
            ax3.tick_params(axis='z', which='major', pad=10)
            ax3.scatter(
                unique_points[:, 0], unique_points[:, 1], unique_points[:, 2], color=colours[2],
                **kwargs)
            ax3.set_xlabel('Easting', fontsize=13, labelpad=15)
            ax3.set_ylabel('Northing', fontsize=13, labelpad=15)
            ax3.set_zlabel('Elevation', fontsize=13, labelpad=12)
            plt.subplots_adjust(left=0, bottom=0, right=0.92, top=1)

        else:
            fig3 = plt.figure(figsize=(11, 5))
            ax3 = fig3.add_subplot(aspect='equal', adjustable='box')
            ax3.scatter(unique_points[:, 0], unique_points[:, 1], color=colours[2], **kwargs)
            ax3.set_xlabel('Easting', fontsize=13, labelpad=8)
            ax3.set_ylabel('Northing', fontsize=13, labelpad=8)
            ax3.grid()

        ax3.ticklabel_format(useOffset=False)

        if add_title:
            ax3.set_title(
                f'Unique points between "{labels[0]}" and "{labels[1]}" entities of polyline\n'
                f'of Tile{tile_name} in {title_date}.')

        fig3.tight_layout()

        self._save_pcd_dgn_shp_view(
            fig3, save_as=save_as, fig_filename_=fig_filename_, dpi=dpi, verbose=verbose)

    def view_pcd_dgn_shp_polyline(self, tile_xy, pcd_date, projection='3d', add_title=False,
                                  save_as=None, dpi=600, verbose=False, **kwargs):
        # noinspection PyShadowingNames
        """
        Visualise polyline data of the DGN-converted shapefile of point cloud data.

        :param tile_xy: X and Y coordinates in reference to a tile for the point cloud data.
        :type tile_xy: tuple | list | str
        :param pcd_date: Date of the point cloud data.
        :type pcd_date: str | int
        :param projection: Projection type of the subplot; defaults to ``'3d'``.
        :type projection: str | None
        :param add_title: Whether to add a title to the plot; defaults to ``False``.
        :type add_title: bool
        :param save_as: File format that the view is saved as; defaults to ``None``.
        :type save_as: str | None
        :param dpi: DPI for saving image; defaults to ``600``.
        :type dpi: int | None
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the function
            `matplotlib.pyplot.scatter`_.

        .. _`matplotlib.pyplot.scatter`:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

        **Examples**::

            >>> from src.shaft import PCDHandler
            >>> from pyhelpers.settings import mpl_preferences
            >>> mpl_preferences(backend='TkAgg')
            >>> pcdh = PCDHandler()
            >>> tile_xy = (340500, 674200)
            >>> pcd_date = '202004'
            >>> # pcdh.view_pcd_dgn_shp_polyline(
            ... #     tile_xy=tile_xy, pcd_date=pcd_date, projection='3d', s=5, save_as=".svg",
            ... #     verbose=True)
            >>> pcdh.view_pcd_dgn_shp_polyline(tile_xy, pcd_date, projection='3d', s=5)

        .. figure:: ../_images/pcdh_view_pcd_dgn_x340500y674200_202004_3d.*
            :name: pcdh_view_pcd_dgn_x340500y674200_202004_3d
            :align: center
            :width: 100%

            DGN-PointCloud data (3D) of Tile (340500, 674200) in April 2020.

        .. figure:: ../_images/pcdh_view_pcd_dgn_x340500y674200_202004_common_3d.*
            :name: pcdh_view_pcd_dgn_x340500y674200_202004_common_3d
            :align: center
            :width: 100%

            Common points (3D) between 'Complex Chain' and 'LineString' entities of polyline
            of Tile (340500, 674200) in April 2020.

        .. figure:: ../_images/pcdh_view_pcd_dgn_x340500y674200_202004_unique_3d.*
            :name: pcdh_view_pcd_dgn_x340500y674200_202004_unique_3d
            :align: center
            :width: 100%

            Unique points (3D) between 'Complex Chain' and 'LineString' entities of polyline
            of Tile (340500, 674200) in April 2020.

        .. code-block:: python

            >>> # 2D plots - Vertical view
            >>> # pcdh.view_pcd_dgn_shp_polyline(
            ... #    tile_xy, pcd_date, projection=None, s=5, save_as=".svg", verbose=True)
            >>> pcdh.view_pcd_dgn_shp_polyline(tile_xy, pcd_date, projection=None, s=5)

        .. figure:: ../_images/pcdh_view_pcd_dgn_x340500y674200_202004.*
            :name: pcdh_view_pcd_dgn_x340500y674200_202004
            :align: center
            :width: 100%

            DGN-PointCloud data of Tile (340500, 674200) in April 2020.

        .. figure:: ../_images/pcdh_view_pcd_dgn_x340500y674200_202004_common.*
            :name: pcdh_view_pcd_dgn_x340500y674200_202004_common
            :align: center
            :width: 100%

            Common points between 'Complex Chain' and 'LineString' entities of polyline
            of Tile (340500, 674200) in April 2020.

        .. figure:: ../_images/pcdh_view_pcd_dgn_x340500y674200_202004_unique.*
            :name: pcdh_view_pcd_dgn_x340500y674200_202004_unique
            :align: center
            :width: 100%

            Unique points between 'Complex Chain' and 'LineString' entities of polyline
            of Tile (340500, 674200) in April 2020.
        """

        _, dgn_pl_pcd = self.load_dgn_shp(pcd_date=pcd_date, layer_name='Polyline')
        dgn_pl_cc_pcd, dgn_pl_ls_pcd = dgn_pl_pcd.values()

        title_date = datetime.datetime.strptime(str(pcd_date), '%Y%m').strftime('%B %Y')

        tile_x, tile_y = get_tile_xy(tile_xy)
        tile_polyline_path = self.get_pcd_tile_mpl_path(tile_xy=(tile_x, tile_y), pcd_date=pcd_date)

        # Get all 'LineString' entities of the DGN data within the example tile
        ls_mask = tile_polyline_path.contains_points(dgn_pl_ls_pcd[:, :2])
        dgn_ls = dgn_pl_ls_pcd[ls_mask]

        # Get all 'Complex Chain' entities of the DGN data within the example tile
        cc_mask = tile_polyline_path.contains_points(dgn_pl_cc_pcd[:, :2])
        dgn_cc = dgn_pl_cc_pcd[cc_mask]

        filename_suffix = '_' + projection if projection else ''
        fig_filename_ = f"pcdh_view_pcd_dgn_x{tile_x}y{tile_y}_{pcd_date}{filename_suffix}"
        tile_name = f'({tile_x}, {tile_y})'

        colours = plt.colormaps.get_cmap('tab10').colors
        labels = ['Complex Chain', 'LineString']

        # Create plots of the sample DGN data
        view_args = dict(
            dgn_ls=dgn_ls, dgn_cc=dgn_cc, projection=projection, colours=colours,
            add_title=add_title, labels=labels, tile_name=tile_name, title_date=title_date,
            save_as=save_as, fig_filename_=fig_filename_, dpi=dpi, verbose=verbose)
        kwargs.update(view_args)

        self._view_pcd_dgn(**kwargs)

        # Common points between 'Complex Chain' and 'LineString' entities
        kwargs.update(
            {'fig_filename_': fig_filename_.replace(f"_{pcd_date}", f"_{pcd_date}_common")})
        self._view_pcd_dgn_common_points(**kwargs)

        # Unique points between 'Complex Chain' and 'LineString' entities
        kwargs.update(
            {'fig_filename_': fig_filename_.replace(f"_{pcd_date}", f"_{pcd_date}_unique")})
        self._view_pcd_dgn_unique_points(**kwargs)
