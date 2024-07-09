"""
This module explores and gathers information on various variables (a.k.a. features)
that influence track fixity.
"""

import copy
import datetime
import functools
import inspect
import sys
import typing

import dateutil.parser
import matplotlib.pyplot as plt
import measurement.measures
import numpy as np
import pandas as pd
import shapely.ops
import shapely.wkt
from pyhelpers._cache import _print_failure_msg
from pyhelpers.geom import find_closest_points
from pyhelpers.ops import np_shift
from pyhelpers.store import save_figure
from pyrcs.converter import mileage_to_yard
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from sklearn.preprocessing import minmax_scale

from src.shaft.movement import TrackMovement
from src.utils.general import TrackFixityDB, cd_docs_source, paired_next
from src.utils.geometry import extrapolate_line_point


class FeatureCollator(TrackMovement):
    """
    Collate data on various features for developing a machine learning model to
    predict track movement (i.e. track fixity parameters).
    """

    def __init__(self, elr=None, db_instance=None, verbose=True, **kwargs):
        """
        :param db_instance: PostgreSQL database instance; defaults to ``None``
        :type db_instance: TrackFixityDB | None
        :param elr: Engineer's Line Reference(s); defaults to ``None``.
        :type elr: str | list | None
        :param verbose: Whether to print relevant information in console; defaults to ``True``.
        :type verbose: bool | int

        :ivar pyhelpers.dbms.PostgreSQL db_instance: PostgreSQL database instance.
        :ivar str | list elr: Engineer's Line Reference(s).

        :ivar Ballast ballast: An instance of :class:`~src.preprocessor.Ballast`.
        :ivar CARRS carrs: An instance of :class:`~src.preprocessor.CARRS`.
        :ivar CNM cnm: An instance of :class:`~src.preprocessor.CNM`.
        :ivar Geology geology: An instance of :class:`~src.preprocessor.Geology`.
        :ivar INM inm: An instance of :class:`~src.preprocessor.INM`.
        :ivar OPAS opas: An instance of :class:`~src.preprocessor.OPAS`.
        :ivar PCD pcd: An instance of :class:`~src.preprocessor.PCD`.
        :ivar Track track: An instance of :class:`~src.preprocessor.Track`.

        :ivar dict pseudo_mileage_dict: Pseudo mileages for all available track IDs;
            defaults to ``None``.
        :ivar pandas.DataFrame waymarks_: Data of waymarks (with pseudo mileages);
            defaults to ``None``.
        :ivar pandas.DataFrame ballast_summary_: Data of ballast summary (with pseudo mileages);
            defaults to ``None``.
        :ivar pandas.DataFrame ballast_features: Features collated from ballast summary
            for the pseudo mileages; defaults to ``None``.
        :ivar pandas.DataFrame structures_: Data of structure presence (with pseudo mileages);
            defaults to ``None``.
        :ivar pandas.DataFrame inm_combined_data_report_:
            INM combined data report (with pseudo mileages); defaults to ``None``.
        :ivar pandas.DataFrame geological_features_: Data of geology (with pseudo mileages);
            defaults to ``None``.
        :ivar pandas.DataFrame track_quality_: Data of track quality (with pseudo mileages);
            defaults to ``None``.

        :ivar pandas.Series subsection_buffer:
            Buffers created for each track subsection; defaults to ``None``.
        :ivar pandas.DataFrame subsection_nodes:
            Nodes (incl. centroid and two ends) for each track subsection; defaults to ``None``.

        :ivar str | list | None element: Element of rail head,
            e.g. left/right top of rail or running edge; defaults to ``None``.
        :ivar str | list | None direction: Railway direction, e.g. up and down directions;
            defaults to ``None``.
        :ivar int | None subsect_len: Length (in metre) of a subsection for which movement
            is calculated; defaults to ``None``.
        :ivar pandas.DataFrame track_movement_: Data of the track movement with collated features.

        **Examples**::

            >>> from src.shaft import FeatureCollator
            >>> fc = FeatureCollator(elr='ECM8')
            Initialising the feature collator ... Done.
            >>> fc.waymarks_
                  ELR  ...                                    pseudo_geometry
            0    ECM8  ...  LINESTRING Z (326301.33949999977 673958.802100...
            1    ECM8  ...  LINESTRING Z (327289.98450000025 674251.033500...
            2    ECM8  ...  LINESTRING Z (327696.2715999996 674194.5047999...
            3    ECM8  ...  LINESTRING Z (328097.8353000004 674240.6223000...
            4    ECM8  ...  LINESTRING Z (328494.3197999997 674324.2128999...
            ..    ...  ...                                                ...
            210  ECM8  ...  LINESTRING Z (396983.12600000016 658556.672000...
            211  ECM8  ...  LINESTRING Z (397172.99619999994 658203.952700...
            212  ECM8  ...  LINESTRING Z (397435.50569999963 657906.502000...
            213  ECM8  ...  LINESTRING Z (397718.7960000001 657613.5120999...
            214  ECM8  ...  LINESTRING Z (397819.31799999997 657439.910499...
            [215 rows x 6 columns]

        **Illustration**::

            import matplotlib.pyplot as plt
            import numpy as np
            from pyhelpers.settings import mpl_preferences

            mpl_preferences(backend='TkAgg', font_name='Times New Roman')

            fig = plt.figure(figsize=(11, 5))
            ax = fig.add_subplot(aspect='equal', adjustable='box')

            for g in fc.waymarks_.pseudo_geometry:
               g = np.array(g.coords)
               ax.plot(g[:, 0], g[:, 1])

            ax.set_xlabel('Easting', fontsize=14)
            ax.set_ylabel('Northing', fontsize=14)

            fig.tight_layout()

            # from pyhelpers.store import save_figure
            # fig_filename = "fc_pseudo_waymarks_demo"
            # save_figure(fig, f"docs\\source\\_images\\{fig_filename}.svg", verbose=True)
            # save_figure(fig, f"docs\\source\\_images\\{fig_filename}.pdf", verbose=True)

        .. figure:: ../_images/fc_pseudo_waymarks_demo.*
            :name: fc_pseudo_waymarks_demo
            :align: center
            :width: 100%

            Pseudo waymarks.
        """

        super().__init__(db_instance=db_instance)

        self.db_instance = TrackFixityDB(verbose=verbose) if db_instance is None else db_instance

        if verbose:
            print("Initialising the feature collator", end=" ... ")

        self.elr = ['ECM7', 'ECM8'] if elr is None else elr

        self.ballast = None
        self.carrs = None
        self.cnm = None
        # self.gpr = None
        self.geology = None
        self.inm = None
        self.opas = None
        # self.pcd = None
        # self.reports20200822 = None
        self.track = None

        try:
            preproc_classes = inspect.getmembers(sys.modules['src.preprocessor'], inspect.isclass)
            for cls_name, preproc_cls in preproc_classes:
                self.__setattr__(cls_name.lower(), preproc_cls(db_instance=self.db_instance))

            self.pseudo_mileage_dict = self._get_pseudo_mileage_dict()

            self.waymarks, self.waymarks_ = self._get_waymarks()

            self.subsection_nodes = None
            self.subsection_buffer_rect = None
            self.subsection_buffer_cir = None

            self.ballast_summary = None
            self.ballast_summary_ = None
            self.ballast_features = None

            self.structures = {}
            self.structures_ = {}

            self.inm_combined_data_report = None
            self.inm_combined_data_report_ = None

            self.geological_features = None
            self.geological_features_ = None

            self.track_quality = None
            self.track_quality_ = None

            self.element = None
            self.direction = None
            self.subsect_len = None

            self.track_movement = None
            self.track_movement_ = None

            if verbose:
                print("Done.")

        except Exception as e:
            _print_failure_msg(e)

    def _get_waymarks(self):
        waymarks = self.cnm.load_waymarks_shp(elr=self.elr, column_names='essential')
        waymarks.rename(columns={'WAYMARK_VALUE': 'Mileage'}, inplace=True)

        waymarks_ = pd.concat(
            [self.cnm.make_pseudo_waymarks(x) for _, x in waymarks.groupby('ELR')])

        return waymarks, waymarks_

    def _get_pseudo_mileage_dict(self):
        return self.__getattribute__('track').load_pseudo_mileage_dict(elr=self.elr)

    def make_subsection_buffer(self, subsect_geom, buf_type=1, buf_dist=None, ret_dat=True):
        """
        Make a buffer for every subsection for which track movement is calculated.

        :param subsect_geom: A sequence of geometry objects each representing a track subsection
        :type subsect_geom: pandas.Series | list | numpy.ndarray | LineString
        :param buf_type: Buffer type; defaults to ``1``;
            options include ``1`` (circle), ``2`` (flat) and ``3`` (square);
            see `shapely.geometry.CAP_STYLE`_ for more details.
        :type buf_type: int
        :param buf_dist: Radius of the buffer; defaults to ``None``;
            see `shapely.geometry.buffer`_ for more details.
        :type buf_dist: int | float | None
        :param ret_dat: Whether to return the processed data; defaults to ``True``.
        :type ret_dat: bool
        :return: Buffer(s) for track subsection(s) for which track movement is calculated.
        :rtype: pandas.Series | list | numpy.ndarray | Polygon

        .. _`shapely.geometry.CAP_STYLE`:
            https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.CAP_STYLE
        .. _`shapely.geometry.buffer`:
            https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.buffer

        **Examples**::

            >>> from src.shaft import FeatureCollator
            >>> fc = FeatureCollator(elr='ECM8')
            Initialising the feature collator ... Done.
            >>> trk_movement = fc.load_movement(
            ...     element='Left Top', direction='Up', subsect_len=1000)
            >>> trk_movement_ = list(trk_movement.values())[0]
            >>> sub_geom = trk_movement_.subsection
            >>> sub_geom
            0     LINESTRING Z (399428.96 653473.9 34.442, 39942...
            1     LINESTRING Z (399611.8618926046 654427.6398435...
            2     LINESTRING Z (399170.8610524134 655321.5851676...
            3     LINESTRING Z (398643.3364543988 656169.1074680...
            4     LINESTRING Z (397987.7488072964 656909.7001183...
                                        ...
            72    LINESTRING Z (344013.9924472624 675799.4646636...
            73    LINESTRING Z (343206.4015371799 675215.9293591...
            74    LINESTRING Z (342238.3969935408 674967.5756095...
            75    LINESTRING Z (341298.8661618733 674644.9047881...
            76    LINESTRING Z (340414.7337679755 674190.3677420...
            Name: subsection, Length: 77, dtype: object
            >>> sub_buffers = fc.make_subsection_buffer(sub_geom, buf_type=1)
            >>> sub_buffers
            0     POLYGON ((399641.74128885 653931.2746024097, 3...
            1     POLYGON ((399432.8613179412 654884.0547386116,...
            2     POLYGON ((398947.8690315923 655758.5577137272,...
            3     POLYGON ((398308.7644842083 656519.5781464423,...
            4     POLYGON ((397864.7254107677 657387.039114261, ...
                                        ...
            72    POLYGON ((343644.6718084301 675485.7349224968,...
            73    POLYGON ((342744.5813225449 675084.2876887596,...
            74    POLYGON ((341772.5156862524 674849.6516215525,...
            75    POLYGON ((340899.879799073 674373.2162596786, ...
            76    POLYGON ((340294.1556065303 674138.0464607034,...
            Name: subsection, Length: 77, dtype: object

        **Illustration**::

            import matplotlib.pyplot as plt
            import numpy as np
            from pyhelpers.settings import mpl_preferences

            mpl_preferences(backend='TkAgg', font_name='Times New Roman')

            fig = plt.figure(figsize=(11, 5))
            ax = fig.add_subplot(aspect='equal', adjustable='box')

            colours = plt.get_cmap('tab10').colors

            for g in sub_geom:
                g = np.array(g.coords)
                ax.plot(g[:, 0], g[:, 1], color=colours[0])
            plt.plot([], [], color=colours[0], label='Track subsections')

            for buf in sub_buffers:
                buf = np.array(buf.exterior.coords)
                ax.plot(buf[:, 0], buf[:, 1], color=colours[1])
            plt.scatter([], [], marker='o', fc='none', ec=colours[1], label='Buffers')

            ax.set_xlabel('Easting', fontsize=14, labelpad=8.0)
            ax.set_ylabel('Northing', fontsize=14, labelpad=8.0)

            ax.legend()
            fig.tight_layout()

            fig.show()

            # from pyhelpers.store import save_figure
            # fig_filename = "fc_make_subsection_buffer_demo"
            # save_figure(fig, f"docs\\source\\_images\\{fig_filename}.svg", verbose=True)
            # save_figure(fig, f"docs\\source\\_images\\{fig_filename}.pdf", verbose=True)

            plt.close(fig)

        .. figure:: ../_images/fc_make_subsection_buffer_demo.*
            :name: fc_make_subsection_buffer_demo
            :align: center
            :width: 100%

            Buffers for track subsections (for which track movement is calculated).
        """

        def _get_subsect_buf(subsect_geom_):
            if buf_type == 1:
                subsect_buf_ = subsect_geom_.interpolate(distance=0.5, normalized=True).buffer(
                    distance=(subsect_geom_.length / 2 if buf_dist is None else buf_dist))
            else:
                buf_dist_ = (100 if buf_type == 2 else 10) if buf_dist is None else buf_dist
                subsect_buf_ = subsect_geom_.buffer(distance=buf_dist_, cap_style=buf_type)
            return subsect_buf_

        if isinstance(subsect_geom, pd.Series):
            subsect_buffer = subsect_geom.map(_get_subsect_buf)
        elif isinstance(subsect_geom, typing.Iterable):
            subsect_buffer = [_get_subsect_buf(subsect_geom_) for subsect_geom_ in subsect_geom]
        else:
            assert isinstance(subsect_geom, LineString)
            subsect_buffer = _get_subsect_buf(subsect_geom)

        if buf_type == 1:
            self.subsection_buffer_cir = subsect_buffer
        elif buf_type == 2:
            self.subsection_buffer_rect = subsect_buffer

        if ret_dat:
            return subsect_buffer

    def view_subsection_buffer(self, buf_type=1, buf_dist=None, fig_size=(11, 5), save_as=None,
                               dpi=600, **kwargs):
        """
        View subsection buffer.

        :param buf_type: Buffer type; defaults to ``1``;
            options include ``1`` (circle), ``2`` (flat) and ``3`` (square);
            see `shapely.geometry.CAP_STYLE`_ for more details.
        :type buf_type: int
        :param buf_dist: Radius of the buffer; defaults to ``None``.;
            see `shapely.geometry.buffer`_ for more details.
        :type buf_dist: int | float | None
        :param fig_size: Figure size; defaults to ``(11, 5)``.
        :type fig_size: tupule
        :param save_as: File format that the view is saved as; defaults to ``None``.
        :type save_as: str | None
        :param dpi: DPI for saving image; defaults to ``600``.
        :type dpi: int | None
        :param kwargs: [Optional] parameters of the method
            :meth:`TrackMovement.load_movement()<src.shaft.TrackMovement.load_movement>`.

        .. _`shapely.geometry.CAP_STYLE`:
            https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.CAP_STYLE
        .. _`shapely.geometry.buffer`:
            https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.buffer

        **Examples**::

            >>> from src.shaft import FeatureCollator
            >>> from pyhelpers.settings import mpl_preferences
            >>> fc = FeatureCollator(elr='ECM8')
            Initialising the feature collator ... Done.
            >>> # mpl_preferences(backend='TkAgg')
            >>> # fc.view_subsection_buffer(
            ... #     element='Left Top', direction='Up', subsect_len=1000, save_as=".svg")
            >>> fc.view_subsection_buffer(element='Left Top', direction='Up', subsect_len=1000)

        .. figure:: ../_images/fc_view_subsection_buffer_demo.*
            :name: fc_view_subsection_buffer_demo
            :align: center
            :width: 100%

            Buffers for ECM8 track subsections of ~1km.
        """

        if self.track_movement is None or kwargs:
            self.track_movement = self.load_movement(**kwargs)

        track_movement = list(self.track_movement.values())[0]

        sub_geom = track_movement.subsection
        sub_buffers = self.make_subsection_buffer(
            subsect_geom=sub_geom, buf_type=buf_type, buf_dist=buf_dist)

        fig = plt.figure(figsize=fig_size, constrained_layout=True)
        ax = fig.add_subplot(aspect='equal', adjustable='box')

        colours = plt.colormaps.get_cmap('tab10').colors

        for g in sub_geom:
            g = np.array(g.coords)
            ax.plot(g[:, 0], g[:, 1], color=colours[0])
        ax.plot([], [], color=colours[0], label='Track subsection')

        for buf in sub_buffers:
            buf = np.array(buf.exterior.coords)
            ax.plot(buf[:, 0], buf[:, 1], color=colours[1])
        ax.scatter([], [], 100, marker='o', fc='none', ec=colours[1], label='Subsection buffer')

        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlabel('Easting', fontsize=18, labelpad=8.0)
        ax.set_ylabel('Northing', fontsize=18, labelpad=8.0)

        ax.legend(loc='best', numpoints=1, ncol=1, fontsize=15)

        # fig.tight_layout()

        if save_as:
            save_fig_args = {'dpi': dpi, 'transparent': True, 'verbose': True}
            for save_as_ in {save_as, ".svg", ".pdf"}:
                path_to_fig = cd_docs_source("_images", "fc_view_subsection_buffer_demo" + save_as_)
                save_figure(fig, path_to_fig, **save_fig_args)

    def get_subsection_centroid_and_ends(self, subsect_geom):
        """
        Get the centroid and two ends of every subsection.

        :param subsect_geom: A sequence of geometry objects each representing a track subsection.
        :type subsect_geom: pandas.Series | list | numpy.ndarray | LineString
        :return: The centroid and two ends of every track subsection.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft import FeatureCollator
            >>> fc = FeatureCollator(elr='ECM8')
            Initialising the feature collator ... Done.
            >>> trk_movement = fc.load_movement(
            ...     element='Left top', direction='Up', subsect_len=1000)
            >>> trk_movement_ = list(trk_movement.values())[0]
            >>> sub_geom = trk_movement_.subsection
            >>> nodes = fc.get_subsection_centroid_and_ends(subsect_geom=sub_geom)
            >>> nodes
                                            subsect_a  ...                        subsect_centroid
            0                 POINT Z (399428.96 6534  ...  POINT Z (399621.74128885 653931.27....
            1   POINT Z (399611.8618926046 654427....  ...  POINT Z (399412.8613179412 654884.0...
            2   POINT Z (399170.8610524134 655321....  ...  POINT Z (398927.8690315923 655758....
            3   POINT Z (398643.3364543988 656169....  ...  POINT Z (398288.7644842083 656519....
            4   POINT Z (397987.7488072964 656909....  ...  POINT Z (397844.7254107677 657387....
            ..                                    ...  ...                                    ...
            72  POINT Z (344013.9924472624 675799....  ...  POINT Z (343624.6718084301 675485....
            73  POINT Z (343206.4015371799 675215....  ...  POINT Z (342724.5813225449 675084....
            74  POINT Z (342238.3969935408 674967....  ...  POINT Z (341752.5156862524 674849....
            75  POINT Z (341298.8661618733 674644....  ...  POINT Z (340879.879799073 674373.2...
            76  POINT Z (340414.7337679755 674190....  ...  POINT Z (340274.1556065303 674138....
            [77 rows x 3 columns]
            >>> nodes.columns.to_list()
            ['subsect_a', 'subsect_b', 'subsect_centroid']
        """

        if not isinstance(subsect_geom, pd.Series):
            subsect_geom_ = pd.Series(subsect_geom, name='subsection')
        else:
            subsect_geom_ = subsect_geom.copy()

        subsect_nodes = pd.DataFrame(
            data=subsect_geom_.map(lambda x: list(x.boundary.geoms)).to_list(),
            columns=['subsect_a', 'subsect_b'])

        subsect_nodes['subsect_centroid'] = subsect_geom_.map(
            lambda x: x.interpolate(0.5, normalized=True))

        self.subsection_nodes = subsect_nodes

        return self.subsection_nodes

    @staticmethod
    def _nearest_waymark_to_subsect_node(subsect_node_geom, waymarks, prefix=''):
        # subsect_nodes = get_subsection_centroid_and_ends(track_movement.subsection)
        # subsect_node_geom = subsect_nodes['subsect_a']

        subsect_cen_arr = np.array([[pt.x, pt.y, pt.z] for pt in subsect_node_geom])

        # Find the nearest waymarks_
        waymarks_pseudo_sect_ends = np.concatenate(
            [np.array(geom.coords) for geom in waymarks.StartMileageGeom])
        waymarks_ckd_tree = cKDTree(waymarks_pseudo_sect_ends)

        _, indices = waymarks_ckd_tree.query(x=subsect_cen_arr, k=1)

        # nearest_waymarks = [Point(waymarks_points[i]) for i in indices]
        essential_columns = ['ELR', 'StartMileage', 'EndMileage', 'pseudo_geometry']
        nearest_wm_ = waymarks.iloc[indices].reset_index(drop=True)[essential_columns]

        if prefix:
            nearest_wm_.columns = [prefix + '_' + x for x in nearest_wm_.columns]

        return nearest_wm_

    @staticmethod
    def _nearest_waymark_to_subsect(x, nearest_waymarks_):
        mileage_columns = [col for col in nearest_waymarks_.columns if col.endswith('Mileage')]
        mileages = np.sort(np.unique([x[y] for y in mileage_columns])).tolist()
        assert isinstance(mileages, list)

        pseudo_geoms = [x[y] for y in nearest_waymarks_.columns if y.endswith('_pseudo_geometry')]
        pseudo_arr = np.unique(
            np.concatenate(list(map(lambda y: np.array(y.coords), pseudo_geoms))), axis=0)

        if pseudo_arr.shape == (3,):
            pseudo_linestring = LineString(np.vstack([pseudo_arr, pseudo_arr]))
        elif pseudo_arr.shape == (1, 3):
            pseudo_linestring = LineString(np.vstack([pseudo_arr[0], pseudo_arr[0]]))
        else:
            pseudo_linestring = LineString(pseudo_arr)

        return mileages[0], mileages[-1], pseudo_linestring

    def find_nearest_waymark_for_subsection(self, subsect_geom):
        """
        Find the nearest waymark for each track subsection.

        :param subsect_geom: A sequence of geometry objects each representing a track subsection.
        :type subsect_geom: pandas.Series | list | numpy.ndarray | LineString
        :return: The nearest waymark for each track subsection.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft import FeatureCollator
            >>> fc = FeatureCollator(elr='ECM8')
            Initialising the feature collator ... Done.
            >>> trk_movement = fc.load_movement(
            ...     element='Left top', direction='Up', subsect_len=1000)
            >>> trk_movement_ = list(trk_movement.values())[0]
            >>> sub_geom = trk_movement_['subsection']
            >>> sub_geom
            0     LINESTRING Z (399428.96 653473.9 34.442, 39942...
            1     LINESTRING Z (399611.8618926046 654427.6398435...
            2     LINESTRING Z (399170.8610524134 655321.5851676...
            3     LINESTRING Z (398643.3364543988 656169.1074680...
            4     LINESTRING Z (397987.7488072964 656909.7001183...
                                        ...
            72    LINESTRING Z (344013.9924472624 675799.4646636...
            73    LINESTRING Z (343206.4015371799 675215.9293591...
            74    LINESTRING Z (342238.3969935408 674967.5756095...
            75    LINESTRING Z (341298.8661618733 674644.9047881...
            76    LINESTRING Z (340414.7337679755 674190.3677420...
            Name: subsection, Length: 77, dtype: object
            >>> nearest_waymark_for_subsect = fc.find_nearest_waymark_for_subsection(sub_geom)
            >>> nearest_waymark_for_subsect
                Waymark_StartMileage  ...                                 Waymark_PseudoGeom
            0                54.1107  ...  LINESTRING Z (397819.318 657439.9104999993 0, ...
            1                54.1107  ...  LINESTRING Z (397819.318 657439.9104999993 0, ...
            2                54.1107  ...  LINESTRING Z (397819.318 657439.9104999993 0, ...
            3                54.1107  ...  LINESTRING Z (397819.318 657439.9104999993 0, ...
            4                54.0440  ...  LINESTRING Z (397435.5056999996 657906.5020000...
            ..                   ...  ...                                                ...
            72               12.0440  ...  LINESTRING Z (343370.9899000004 675301.2421000...
            73               11.0880  ...  LINESTRING Z (342214.7589999996 674971.2629000...
            74               11.0000  ...  LINESTRING Z (341443.9992000004 674738.0125999...
            75               10.0440  ...  LINESTRING Z (340382.7313999999 674179.2847000...
            76               10.0000  ...  LINESTRING Z (339999.9721999997 674043.0051000...
            [77 rows x 3 columns]
        """

        subsect_nodes = self.get_subsection_centroid_and_ends(subsect_geom)

        waymarks = self.waymarks_.copy()

        nearest_waymarks = [
            self._nearest_waymark_to_subsect_node(subsect_nodes[col], waymarks, prefix=col)
            for col in subsect_nodes.columns]

        nearest_waymarks_ = pd.concat(nearest_waymarks, axis=1)

        temp = nearest_waymarks_.apply(
            functools.partial(
                self._nearest_waymark_to_subsect, nearest_waymarks_=nearest_waymarks_),
            axis=1)
        nearest_waymarks = pd.DataFrame(
            temp.tolist(),
            columns=['Waymark_StartMileage', 'Waymark_EndMileage', 'Waymark_PseudoGeom'])

        return nearest_waymarks

    @staticmethod
    def make_mileage_sequence(dat, start_mil_col, end_mil_col=None):
        """
        Put together all values of the mileages into a sequence.

        :param dat: A data set in which mileage data (of both start and end) is available.
        :type dat: pandas.DataFrame
        :param start_mil_col: Column name of the start mileage data.
        :type start_mil_col: str
        :param end_mil_col: Column name of the end mileage data; defaults to ``None``.
        :type end_mil_col: str | None
        :return: A sequence of mileages in the given data set ``dat``,
            and distances (in metres) between adjacent mileages.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft import FeatureCollator
            >>> test_elr = 'ECM8'
            >>> fc = FeatureCollator(elr=test_elr)
            Initialising the feature collator ... Done.
            >>> ballast_summary = fc.ballast.load_data(elr=test_elr)
            >>> mil_seq = fc.make_mileage_sequence(ballast_summary, 'StartMileage')
            >>> mil_seq
                  Mileage  metre_to_next
            0      0.0618         9.1440
            1      0.0628         1.8288
            2      0.0630        17.3736
            3      0.0649         8.2296
            4      0.0658         1.8288
                   ...            ...
            2851  54.1100         0.9144
            2852  54.1101         0.9144
            2853  54.1102         1.8288
            2854  54.1104         2.7432
            2855  54.1107         0.0000
            [2856 rows x 2 columns]
            >>> mil_seq = fc.make_mileage_sequence(ballast_summary, 'StartMileage', 'EndMileage')
            >>> mil_seq
                  Mileage  metre_to_next
            0      0.0618         9.1440
            1      0.0628         1.8288
            2      0.0630        17.3736
            3      0.0649         8.2296
            4      0.0658         1.8288
                   ...            ...
            2852  54.1101         0.9144
            2853  54.1102         1.8288
            2854  54.1104         2.7432
            2855  54.1107         0.9144
            2856  54.1108         0.0000
            [2857 rows x 2 columns]
        """

        if end_mil_col is None:
            temp = dat[[start_mil_col]].values
        else:
            temp = dat[[start_mil_col, end_mil_col]].values

        mil_val_ = np.sort(np.unique(np.concatenate(temp)))
        yard_val = np.vectorize(mileage_to_yard)(mil_val_)
        next_yard_val = np_shift(array=yard_val, step=-1, fill_value=yard_val[-1])
        metre_to_next = [measurement.measures.Distance(yd=x).m for x in (next_yard_val - yard_val)]

        mileage_seq = pd.DataFrame({'Mileage': mil_val_, 'metre_to_next': metre_to_next})

        return mileage_seq

    @staticmethod
    def _fill_na_with_waymarks(mileage_geom):
        mileage_geom_ = mileage_geom.copy()

        indices = mileage_geom_['pseudo_geometry'].dropna().index.to_list()

        if len(indices) > 1:
            pseudo_geom = pd.Series(
                index=mileage_geom_.index, dtype=mileage_geom_.pseudo_geometry.dtype)
            for i, j in list(paired_next(indices)):
                sect_ls = LineString(
                    [mileage_geom_.pseudo_geometry.loc[i], mileage_geom_.pseudo_geometry.loc[j]])

                dist = mileage_geom_['metre_to_next'].loc[i:j - 1]
                cum_dist = np.cumsum(dist.values)

                if len(dist) > 2:
                    alpha = 0.99
                    while cum_dist.max() > sect_ls.length:
                        cum_dist *= alpha
                        alpha -= 0.01
                    cum_dist = minmax_scale(
                        cum_dist, feature_range=(cum_dist.min(), sect_ls.length))

                for k in range(i, j - 1):
                    pseudo_geom.loc[k + 1] = sect_ls.interpolate(distance=cum_dist[k - i])

            mileage_geom_['pseudo_geometry'] = \
                mileage_geom_['pseudo_geometry'].combine_first(pseudo_geom)

        return mileage_geom_

    @staticmethod
    def _fill_pseudo_first_last_na(mileage_geom):
        mileage_geom_ = mileage_geom.copy()

        first_non_nan_idx = mileage_geom_['pseudo_geometry'].first_valid_index()
        if first_non_nan_idx != mileage_geom_.index[0]:
            for i in range(first_non_nan_idx, mileage_geom_.index[0], -1):
                pseudo_ls = LineString(mileage_geom_['pseudo_geometry'].loc[i:i + 2].to_list())
                p_dist = mileage_geom_.metre_to_next.loc[i - 1]
                prev_p = extrapolate_line_point(
                    polyline=pseudo_ls, dist=p_dist, reverse=True, as_geom=False)
                mileage_geom_.loc[i - 1, 'pseudo_geometry'] = Point(
                    np.append(np.array(prev_p), 0.0))

        last_non_nan_idx = mileage_geom_['pseudo_geometry'].last_valid_index()
        if last_non_nan_idx != mileage_geom_.index[-1]:
            for i in range(last_non_nan_idx, mileage_geom_.index[-1]):
                pseudo_ls = LineString(mileage_geom_['pseudo_geometry'].loc[i - 1:i].to_list())
                p_dist = mileage_geom_['metre_to_next'].loc[i]
                next_p = extrapolate_line_point(polyline=pseudo_ls, dist=p_dist, as_geom=False)
                mileage_geom_.loc[i + 1, 'pseudo_geometry'] = Point(np.append(np.array(next_p), 0))

        return mileage_geom_

    def assign_pseudo_mileage(self, elr, data, tid_col='TID', start_mil_col='StartMileage',
                              end_mil_col='EndMileage', pseudo_mil_dict=None, suppl_mil_data=None,
                              suppl_mil_col='Mileage'):
        """
        Assign pseudo mileage to cases in a given data set.

        :param elr: Engineer's Line Reference.
        :type elr: str
        :param data: A data set of one resource of the project.
        :type data: pandas.DataFrame
        :param tid_col: Name of the column for track IDs; defaults to ``'TID'``.
        :type tid_col: str
        :param start_mil_col: Name of the column for start mileage; defaults to ``'StartMileage'``.
        :type start_mil_col: str
        :param end_mil_col: Name of the column for end mileage; defaults to ``'EndMileage'``.
        :type end_mil_col: str
        :param suppl_mil_data: A supplementary data set about mileages; defaults to ``None``.
        :type suppl_mil_data: dict | pandas.DataFrame | None
        :param suppl_mil_col: Column name for the mileage data in ``suppl_mil_data``;
            defaults to ``'Mileage'``.
        :type suppl_mil_col: str
        :param pseudo_mil_dict: Data of pseudo mileages;
            when ``pseudo_mil_dict=None`` (default), it retrieves the data by using the method
            :meth:`Track.load_pseudo_mileage_dict()
            <src.preprocessor.Track.load_pseudo_mileage_dict>`.
        :type pseudo_mil_dict: dict | None
        :return: Processed data set with assigned pseudo mileages.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft import FeatureCollator
            >>> test_elr = 'ECM8'
            >>> fc = FeatureCollator(elr=test_elr)
            Initialising the feature collator ... Done.
            >>> ballast_summary = fc.ballast.load_data(elr=test_elr)
            >>> ballast_summary.head()
                   ID  GEOGIS Switch ID Track priority  ...   BCF StartMileage EndMileage
            0  225960               NaN  Running lines  ...  1.50       0.1252     0.1291
            1  225961           23481.0  Running lines  ...  1.00       0.1291     0.1320
            2  225962           23481.0  Running lines  ...  1.00       0.1320     0.1350
            3  225963               NaN  Running lines  ...  1.47       0.1350     0.1367
            4  225964               NaN  Running lines  ...  1.03       0.1367     0.1368
            [5 rows x 60 columns]
            >>> ballast_summary_ = fc.assign_pseudo_mileage(elr=test_elr, data=ballast_summary)
            >>> ballast_summary_.head()
                   ID  ...                                    pseudo_geometry
            0  225960  ...  LINESTRING Z (326843.3266799995 674149.1670249...
            1  225961  ...  LINESTRING Z (326876.7641107336 674161.6463143...
            2  225962  ...  LINESTRING Z (326901.8077764853 674171.0149599...
            3  225963  ...  LINESTRING Z (326927.8255407451 674180.4099819...
            4  225964  ...  LINESTRING Z (326942.5269018637 674185.8494323...
            [5 rows x 63 columns]

        **Illustration**::

            import numpy as np
            import matplotlib.pyplot as plt
            from pyhelpers.settings import mpl_preferences

            mpl_preferences(backend='TkAgg', font_name='Times New Roman')

            fig = plt.figure(figsize=(11, 5))
            ax = fig.add_subplot(aspect='equal', adjustable='box')

            for g in ballast_summary_.pseudo_geometry:
                g_ = np.array(g.coords)
                ax.plot(g_[:, 0], g_[:, 1])

            ax.set_xlabel('Easting', fontsize=14, labelpad=10)
            ax.set_ylabel('Northing', fontsize=14, labelpad=10)

            fig.tight_layout()

            # from pyhelpers.store import save_figure
            # fig_filename = "fc_assign_pseudo_mileage_demo"
            # save_figure(fig, f"docs\\source\\_images\\{fig_filename}.svg", verbose=True)
            # save_figure(fig, f"docs\\source\\_images\\{fig_filename}.pdf", verbose=True)

        .. figure:: ../_images/fc_assign_pseudo_mileage_demo.*
            :name: fc_assign_pseudo_mileage_demo
            :align: center
            :width: 100%

            Pseudo mileages for a subset of the data of ECM8 ballast summary.
        """

        assert all(x in data.columns for x in [start_mil_col, end_mil_col] if x is not None)

        if suppl_mil_data is None:
            waymarks_ = self.waymarks.query(f'ELR == "{elr}"').drop(columns=['ELR'])
            ref_mil_col_ = 'Mileage'
            ref_mil_dat = waymarks_.set_index(ref_mil_col_)
        else:
            ref_mil_dat = suppl_mil_data[elr]
            ref_mil_col_ = copy.copy(suppl_mil_col)

        if pseudo_mil_dict is None:
            if self.pseudo_mileage_dict is None:
                pseudo_trk_shp_mileage_dict = self.track.load_pseudo_mileage_dict(elr=elr)
            else:
                pseudo_trk_shp_mileage_dict = self.pseudo_mileage_dict.copy()
        else:
            pseudo_trk_shp_mileage_dict = copy.copy(pseudo_mil_dict)

        processed_data_list = []

        for tid, dat in data.groupby(tid_col):
            mileage_seq = self.make_mileage_sequence(dat, start_mil_col, end_mil_col)

            try:  # if tid in pseudo_trk_shp_mileage_dict.keys():
                pseudo_trk_shp_mileage = pseudo_trk_shp_mileage_dict[elr][tid]
                mileage_geom = mileage_seq.join(pseudo_trk_shp_mileage, on='Mileage')

                nan_idx = np.where(mileage_geom['pseudo_geometry'].isna())[0]
                if len(nan_idx) > 0:
                    temp = mileage_geom.copy()
                    temp['Mile'] = temp['Mileage'].astype(int)
                    temp_subs = dict(list(temp.groupby('Mile')))
                    for mil, sd in temp_subs.items():
                        if set(sd.index) & set(nan_idx):
                            if len(sd) > 1:
                                sd = self._fill_na_with_waymarks(sd)
                                try:
                                    temp_subs[mil] = self._fill_pseudo_first_last_na(sd)
                                except (AttributeError, TypeError):
                                    val = sd['pseudo_geometry'].loc[
                                        sd['pseudo_geometry'].first_valid_index()]
                                    temp_subs[mil]['pseudo_geometry'] = val
                    mileage_geom = pd.concat(temp_subs.values()).drop(columns=['Mile'])

                # fig = plt.figure()
                # ax = fig.add_subplot()
                # for p in mileage_geom.pseudo_geometry:
                #     ax.scatter(p.x, p.y)

            except KeyError:  # else:
                if isinstance(ref_mil_dat, dict):
                    try:
                        ref_mil_dat_ = ref_mil_dat[tid]
                    except KeyError:
                        waymarks_ = self.waymarks.query(f'ELR == "{elr}"').drop(columns=['ELR'])
                        ref_mil_dat_ = waymarks_.set_index('Mileage')
                    mileage_geom = mileage_seq.join(ref_mil_dat_, on=ref_mil_col_)
                else:
                    mileage_geom = mileage_seq.join(ref_mil_dat, on=ref_mil_col_)
                mileage_geom.rename(columns={'geometry': 'pseudo_geometry'}, inplace=True)

                mileage_geom = self._fill_na_with_waymarks(mileage_geom)

            if mileage_geom['pseudo_geometry'].any():
                mileage_geom_ = self._fill_pseudo_first_last_na(mileage_geom)

                pseudo_mileage_geom = mileage_geom_[['Mileage', 'pseudo_geometry']].set_index(
                    'Mileage')

                dat_ = dat.join(pseudo_mileage_geom, on=start_mil_col)
                dat_ = dat_.join(
                    pseudo_mileage_geom, on=end_mil_col, lsuffix='_start', rsuffix='_end')
                dat_['pseudo_geometry'] = dat_.apply(
                    lambda x: [x.pseudo_geometry_start, x.pseudo_geometry_end], axis=1)
                dat_['pseudo_geometry'] = dat_['pseudo_geometry'].map(LineString)

                # fig = plt.figure()
                # ax = fig.add_subplot()
                # for g in dat_.pseudo_geometry:
                #     g_arr = np.array(g)
                #     ax.plot(g_arr[:, 0], g_arr[:, 1])

                processed_data_list.append(dat_)

        processed_data = pd.concat(processed_data_list)

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # for g in processed_data.pseudo_geometry:
        #     g_arr = np.array(g)
        #     ax.plot(g_arr[:, 0], g_arr[:, 1])

        return processed_data

    # == Ballast ===================================================================================
    def assign_pseudo_mileage_to_ballast(self, elr=None, pseudo_mil_dict=None, ret_dat=False):
        """
        Assign pseudo mileage to the data of ballast summary.

        :param elr: Engineer's Line Reference(s); defaults to ``None``.
        :type elr: str | list | None
        :param pseudo_mil_dict: Data of pseudo mileages;
            when ``pseudo_mil_dict=None`` (default), it retrieves the data by using the method
            :meth:`Track.load_pseudo_mileage_dict()
            <src.preprocessor.Track.load_pseudo_mileage_dict>`.
        :type pseudo_mil_dict: dict | None
        :param ret_dat: Whether to return the processed data; defaults to ``False``.
        :type ret_dat: bool
        :return: Data of ballast summary with assigned pseudo mileages.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft import FeatureCollator
            >>> fc = FeatureCollator(elr=['ECM7', 'ECM8'])
            Initialising the feature collator ... Done.
            >>> fc.assign_pseudo_mileage_to_ballast()
            >>> fc.ballast_summary
                      ID  GEOGIS Switch ID Track priority  ...   BCF StartMileage EndMileage
            0     221491               NaN  Running lines  ...  1.27       0.0000     0.0054
            1     221492               NaN  Running lines  ...  1.27       0.0054     0.0060
            2     221493               NaN  Running lines  ...  1.27       0.0060     0.0066
            3     221494               NaN  Running lines  ...  1.28       0.0066     0.0084
            4     221495               NaN  Running lines  ...  1.28       0.0084     0.0092
                  ...               ...            ...  ...   ...          ...        ...
            8179  230311               NaN  Running lines  ...  1.29       3.0694     3.0748
            8180  230312               NaN  Running lines  ...  1.29       3.0748     3.0749
            8181  230313               NaN  Running lines  ...  1.29       3.0749     3.0777
            8182  230314               NaN  Running lines  ...  1.29       3.0777     3.0832
            8183  230315           24283.0  Running lines  ...  1.20       3.0832     3.0869
            [8184 rows x 60 columns]
            >>> fc.ballast_summary_
                      ID  ...                                    pseudo_geometry
            0     221491  ...  LINESTRING Z (424617.6578774111 563817.2014027...
            1     221492  ...  LINESTRING Z (424666.8570424306 563821.3953917...
            2     221493  ...  LINESTRING Z (424672.446802165 563821.87189151...
            3     221494  ...  LINESTRING Z (424678.0365618993 563822.3483912...
            4     221495  ...  LINESTRING Z (424694.8074982702 563823.7573691...
                  ...  ...                                                ...
            8179  230311  ...  LINESTRING Z (330859.5571851924 673253.5898562...
            8180  230312  ...  LINESTRING Z (330907.5838895333 673241.5453524...
            8181  230313  ...  LINESTRING Z (330908.473272947 673241.32230603...
            8182  230314  ...  LINESTRING Z (330933.3171053747 673234.8605545...
            8183  230315  ...  LINESTRING Z (330981.5182506089 673220.0618065...
            [8180 rows x 63 columns]
            >>> fc.ballast_summary_[fc.ballast_summary_.columns[-3:]]
                                pseudo_geometry_start  ...                         pseudo_geometry
            0     POINT Z (424617.6578774138 563817  ...  LINESTRING Z (424617.6578774138 56381...
            1     POINT Z (424666.8570424306 563821  ...  LINESTRING Z (424666.8570424306 56382...
            2      POINT Z (424672.446802165 563821  ...  LINESTRING Z (424672.446802165 563821...
            3     POINT Z (424678.0365618993 563822  ...  LINESTRING Z (424678.0365618993 56382...
            4     POINT Z (424694.8074982702 563823  ...  LINESTRING Z (424694.8074982702 56382...
                                                   ...  ...                                    ...
            8179  POINT Z (330859.5571851924 673253  ...  LINESTRING Z (330859.5571851924 67325...
            8180  POINT Z (330907.5838895333 673241  ...  LINESTRING Z (330907.5838895333 67324...
            8181    POINT Z (330908.473272947 67324  ...  LINESTRING Z (330908.473272947 673241...
            8182  POINT Z (330933.3171053747 673234  ...  LINESTRING Z (330933.3171053747 67323...
            8183  POINT Z (330981.5182506089 673220  ...  LINESTRING Z (330981.5182506089 67322...
            [8180 rows x 3 columns]

        .. seealso::

            - Examples for the method :meth:`~src.modeller.FeatureCollator.assign_pseudo_mileage`.
        """

        self.ballast_summary = self.ballast.load_data(elr=self.elr if elr is None else elr)

        # # Debugging
        # for k, v in list(ballast_.groupby('ELR')):
        #     try:
        #         temp_ = self.assign_pseudo_mileage(k, v)
        #     except Exception as e:
        #         print(e, k)
        #         break

        ballast_dat_list = [
            self.assign_pseudo_mileage(elr=k, data=v, pseudo_mil_dict=pseudo_mil_dict)
            for k, v in list(self.ballast_summary.groupby('ELR'))
        ]

        self.ballast_summary_ = pd.concat(ballast_dat_list, axis=0).sort_index()

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # for g in ballast.pseudo_geometry:
        #     g_arr = np.array(g)
        #     ax.plot(g_arr[:, 0], g_arr[:, 1])

        if ret_dat:
            return self.ballast_summary_

    def collate_ballast_features(self, elr=None, ret_dat=False, **kwargs):
        """
        Collate feature data from the data of ballast summary (for modelling).

        :param elr: Engineer's Line Reference(s); defaults to ``None``.
        :type elr: str | list | None
        :param ret_dat: Whether to return the processed data; defaults to ``False``.
        :type ret_dat: bool
        :param kwargs: [Optional] parameters of the method
            :meth:`TrackMovement.load_movement()<src.shaft.movement.TrackMovement.load_movement>`.
        :return: Data (basic statistics) of ballast summary for modelling.
        :rtype: pandas.DataFrame

        .. attention::

            Currently, the features include only curvature, cant, maximum speed and
            maximum axle load.

        **Examples**::

            >>> from src.shaft import FeatureCollator
            >>> fc = FeatureCollator(elr=['ECM7', 'ECM8'])
            >>> fc.collate_ballast_features(element='Left Top', direction='Up', subsect_len=1000)
            >>> fc.ballast_features
                                                                Curvature  ...  Max axle load
            subsection                                                     ...
            LINESTRING Z (399428.96 653473.9 34.442, 399429...  -0.000849  ...             26
            LINESTRING Z (399611.8618926046 654427.63984359...  -0.000849  ...             26
            LINESTRING Z (399170.8610524134 655321.58516764...  -0.000337  ...             26
            LINESTRING Z (398643.3364543988 656169.10746808...   0.001189  ...             26
            LINESTRING Z (397987.7488072964 656909.70011834...  -0.001266  ...             26
                                                                   ...  ...            ...
            LINESTRING Z (344013.9924472624 675799.46466366...   0.001075  ...             26
            LINESTRING Z (343206.4015371799 675215.92935916...   0.001075  ...             26
            LINESTRING Z (342238.3969935408 674967.57560955...  -0.000416  ...             26
            LINESTRING Z (341298.8661618733 674644.90478813...   0.000898  ...             26
            LINESTRING Z (340414.7337679755 674190.36774200...   0.000000  ...             26
            [77 rows x 4 columns]
            >>> fc.ballast_features.index.name
            'subsection'
            >>> fc.ballast_features.columns.to_list()
            ['Curvature', 'Cant', 'Max speed', 'Max axle load']
        """

        if self.track_movement is None or kwargs:
            self.track_movement = self.load_movement(**kwargs)
        track_movement = list(self.track_movement.values())[0]

        if self.ballast_summary_ is None:
            self.assign_pseudo_mileage_to_ballast(elr=self.elr if elr is None else elr)
        ballast_summary_ = self.ballast_summary_.copy()

        if self.subsection_buffer_rect is None:
            self.make_subsection_buffer(
                subsect_geom=track_movement.subsection, buf_type=2, buf_dist=1.5)

        buf_cen_arr = np.vstack(
            [np.array(buf.centroid.coords) for buf in self.subsection_buffer_rect])
        ballast_sect_cen_arr = np.vstack(
            [np.array(ls.centroid.coords[0]) for ls in ballast_summary_.pseudo_geometry])

        _, bal_indices = find_closest_points(
            pts=buf_cen_arr, ref_pts=ballast_sect_cen_arr, k=100, ret_idx=True)

        ballast_indices = []
        for subsect_buf, bal_idx in zip(self.subsection_buffer_rect, bal_indices):
            ballast_idx = []
            for i in bal_idx:
                bal_pseudo_geom = ballast_summary_.pseudo_geometry.loc[i]
                if subsect_buf.intersects(bal_pseudo_geom) or subsect_buf.contains(bal_pseudo_geom):
                    ballast_idx.append(i)
            ballast_indices.append(ballast_idx)

        feature_columns = ['Curvature', 'Cant', 'Max speed', 'Max axle load']
        feature_calc = dict(
            zip(feature_columns, [lambda x: max(x, key=abs)] * len(feature_columns)))

        bal_stats_dat = []
        for idx in self.subsection_buffer_rect.index:
            bal_dat = ballast_summary_.loc[ballast_indices[idx]]
            if bal_dat.empty:
                bal_stats = pd.DataFrame(
                    dict(zip(feature_columns, [np.nan] * len(feature_columns))), index=[idx])
            else:
                bal_stats = bal_dat.groupby([True] * len(bal_dat)).agg(feature_calc)
                bal_stats.index = [idx]
            bal_stats_dat.append(bal_stats)

        ballast_features_ = pd.concat(bal_stats_dat, axis=0)
        ballast_features_.index = track_movement.subsection
        self.ballast_features = ballast_features_

        if ret_dat:
            return self.ballast_features

    # == CARRS and OPAS ============================================================================
    def collate_nearest_structure(self, structure_data, structure_name, ret_dat=False, **kwargs):
        """
        Collate information about the presence of a certain structure at each subsection
        (for which track movement is calculated).

        :param structure_data: Data of a structure.
        :type structure_data: pandas.DataFrame
        :param structure_name: Name of a structure.
        :type structure_name: str
        :param ret_dat: Whether to return the processed data; defaults to ``False``.
        :type ret_dat: bool
        :param kwargs: [Optional] parameters of the method
            :meth:`TrackMovement.load_movement()<src.shaft.movement.TrackMovement.load_movement>`.
        :return: Count of the nearest asset to each subsection
            for which track movement is calculated.
        :rtype: dict

        **Examples**::

            >>> from src.shaft import FeatureCollator
            >>> test_elr = 'ECM8'
            >>> fc = FeatureCollator(elr=test_elr)
            Initialising the feature collator ... Done.
            >>> fc.track_movement = fc.load_movement(
            ...     element='Left Top', direction='Up', subsect_len=1000)
            >>> type(fc.track_movement)
            dict
            >>> list(fc.track_movement.keys())
            ['Up_LeftTopOfRail_201910_202004']
            >>> trk_movement_ = fc.track_movement['Up_LeftTopOfRail_201910_202004']
            >>> trk_movement_.head()
                                                  subsection  ...  vertical_displacement_b_abs_max
            0  LINESTRING Z (399428.96 653473.9 34.442, 3...  ...                        -0.010023
            1  LINESTRING Z (399611.8618926046 654427.639...  ...                        -0.006940
            2  LINESTRING Z (399170.8610524134 655321.585...  ...                        -0.002362
            3  LINESTRING Z (398643.3364543988 656169.107...  ...                        -0.010806
            4  LINESTRING Z (397987.7488072964 656909.700...  ...                        -0.007337
            [5 rows x 37 columns]
            >>> ol_bdg = fc.carrs.load_overline_bridges_shp(elr=test_elr)
            >>> ol_bdg.head()
              Entity  ...                                         geometry
            0  Point  ...  POINT Z (329001.3986999998 674251.2369999997 0)
            1  Point  ...  POINT Z (329243.2949999999 674171.7059000004 0)
            2  Point  ...  POINT Z (331516.5432000002 673158.5880999994 0)
            3  Point  ...  POINT Z (331284.2423999999 673210.6504999995 0)
            4  Point  ...  POINT Z (331820.4682999998 673006.2052999996 0)
            [5 rows x 78 columns]
            >>> fc.collate_nearest_structure(ol_bdg, structure_name='Overline bridges')
            >>> fc.structures_['Overline bridges'].head()
                                                                Overline bridges
            subsection
            LINESTRING Z (399428.96 653473.9 34.442, 399429...                 0
            LINESTRING Z (399611.8618926046 654427.63984359...                 0
            LINESTRING Z (399170.8610524134 655321.58516764...                 0
            LINESTRING Z (398643.3364543988 656169.10746808...                 0
            LINESTRING Z (397987.7488072964 656909.70011834...                 0
            >>> fc.structures_['Overline bridges'].sum()
            Overline bridges    40
            dtype: int64

        **Illustration**::

            import numpy as np
            import matplotlib.pyplot as plt
            from shapely.geometry import MultiPolygon
            from pyhelpers.settings import mpl_preferences

            mpl_preferences(backend='TkAgg', font_name='Times New Roman')

            # Get track geometry
            track_movement = list(fc.track_movement.values())[0]
            sub_geom = track_movement.subsection
            sub_buffers = fc.make_subsection_buffer(sub_geom, buf_type=1, buf_dist=None)

            fig = plt.figure(figsize=(11, 5), constrained_layout=True)
            ax = fig.add_subplot(aspect='equal', adjustable='box')

            colours = plt.get_cmap('tab10').colors

            for g in sub_geom:
                g = np.array(g.coords)
                ax.plot(g[:, 0], g[:, 1], color=colours[0])
            ax.plot([], [], color=colours[0], label='Track section (length = 1 km)')

            for buf in fc.subsection_buffer_cir:
                buf_ = np.array(buf.exterior.coords)
                ax.plot(buf_[:, 0], buf_[:, 1], color=colours[1])
            ax.scatter(
                [], [], 100, marker='o', fc='none', ec=colours[1],
                label='Buffer zone (diameter = 1 km)')

            # Select all overline bridges that are within the study area
            convex_hull = MultiPolygon(fc.subsection_buffer_cir.to_list()).convex_hull
            ol_bdg_geometry = [x for x in ol_bdg.geometry if x.within(convex_hull)]

            for ob in ol_bdg_geometry:
                ax.scatter(ob.x, ob.y, color=colours[2], s=25)
            ax.scatter([], [], marker='o', color=colours[2], label='Overline bridge')

            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.set_xlabel('Easting', fontsize=20, labelpad=8.0)
            ax.set_ylabel('Northing', fontsize=20, labelpad=8.0)

            ax.legend(loc='best', numpoints=1, ncol=1, fontsize=18)

            # from pyhelpers.store import save_figure
            # fig_pathname = "docs/source/_images/fc_collate_nearest_structure_demo"
            # save_figure(fig, f"{fig_pathname}.svg", transparent=True, verbose=True)
            # save_figure(fig, f"{fig_pathname}.pdf", transparent=True, verbose=True)

        .. figure:: ../_images/fc_collate_nearest_structure_demo.*
            :name: fc_collate_nearest_structure_demo
            :align: center
            :width: 100%

            Overline bridges and buffers of the track subsections.
        """

        if self.track_movement is None or kwargs:
            self.track_movement = self.load_movement(**kwargs)
        track_movement = list(self.track_movement.values())[0]

        if self.subsection_buffer_cir is None:
            self.make_subsection_buffer(
                subsect_geom=track_movement.subsection, buf_type=1, ret_dat=False)

        indices = [
            [i for i, a in enumerate(structure_data.geometry) if buf.contains(a)]
            for buf in self.subsection_buffer_cir]
        indices_count = pd.Series([len(i) for i in indices], name=structure_name)

        structure_count = pd.concat(
            [track_movement.subsection, indices_count], axis=1).set_index('subsection')

        structure_count_ = {structure_name: structure_count}
        self.structures_.update(structure_count_)

        self.structures.update({structure_name: structure_data})

        if ret_dat:
            return structure_count_

    # == INM =======================================================================================
    @staticmethod
    def _add_pseudo_mileage_range(numbered_mil_val, mil_data, s_mil_col, e_mil_col,
                                  col_name='pseudo_range'):
        tmp = mil_data.join(numbered_mil_val, on=s_mil_col)
        tmp = tmp.join(numbered_mil_val, on=e_mil_col, rsuffix='_end')

        tmp[col_name] = tmp.apply(lambda x: range(x.tmp, x.tmp_end + 1), axis=1)

        mileage_dat = tmp.drop(columns=['tmp', 'tmp_end'])

        return mileage_dat

    @staticmethod
    def _make_pseudo_geom(trk_mileages_, x_elr, x_tid, x_mil_range):
        # trk_mil = trk_mileages_.query('ELR == \'{}\' and TRID == {}'.format(x_elr, x_tid))
        trk_mil = trk_mileages_[(trk_mileages_['ELR'] == x_elr) & (trk_mileages_['TRID'] == x_tid)]

        idx = [i for i, x in enumerate(trk_mil.pseudo_range) if set(x_mil_range) & set(x)]

        geom = trk_mileages_.loc[idx, 'geometry'].to_list()
        if len(geom) == 1:
            geom_ = geom[0]
        else:
            geom_ = shapely.ops.linemerge(geom)

        return geom_

    def _find_nearest_trk_shp_for_inm_cdr(self, elr=None, pseudo_range_col_name='pseudo_range'):
        # elr = ['ECM7', 'ECM8']
        elr_ = copy.copy(self.elr) if elr is None else copy.copy(elr)

        inm_cdr = self.inm.load_combined_data_report(elr=elr_)
        tracks_shp = self.track.load_tracks_shp(elr=elr_)

        inm_mileage_columns = ['ELR', 'ELR_STARTMEASURE', 'ELR_ENDMEASURE', 'REF_TRACKID']
        inm_cdr_mileages = inm_cdr[inm_mileage_columns]

        trk_mileage_columns = ['ELR', 'L_M_FROM', 'L_M_TO', 'TRID', 'geometry']
        trk_mileages = tracks_shp[trk_mileage_columns]

        mixed_mil_val = np.unique(
            np.concatenate([
                inm_cdr_mileages['ELR_STARTMEASURE'].values,
                inm_cdr_mileages['ELR_ENDMEASURE'].values,
                trk_mileages['L_M_FROM'].values,
                trk_mileages['L_M_TO'].values]))

        numbered_mil_val = pd.Series(
            range(len(mixed_mil_val)), index=np.sort(mixed_mil_val), name='tmp')

        inm_mileages_ = self._add_pseudo_mileage_range(
            numbered_mil_val, inm_cdr_mileages, 'ELR_STARTMEASURE', 'ELR_ENDMEASURE',
            col_name=pseudo_range_col_name)
        trk_mileages_ = self._add_pseudo_mileage_range(
            numbered_mil_val, trk_mileages, 'L_M_FROM', 'L_M_TO', col_name=pseudo_range_col_name)

        trk_overlap = inm_mileages_.apply(
            lambda x: self._make_pseudo_geom(
                trk_mileages_, x['ELR'], x['REF_TRACKID'], x[pseudo_range_col_name]),
            axis=1)

        inm_cdr['overlap_trk_shp'] = trk_overlap

        return inm_cdr

    def assign_pseudo_mileage_to_inm_cdr(self, elr=None, pseudo_mil_dict=None, ret_dat=False):
        """
        Assign pseudo mileage to the INM combined data report.

        :param elr: Engineer's Line Reference(s); defaults to ``None``.
        :type elr: str | list | None
        :param pseudo_mil_dict: Data of pseudo mileages;
            when ``pseudo_mil_dict=None`` (default), it retrieves the data by using the method
            :meth:`Track.load_pseudo_mileage_dict()
            <src.preprocessor.track.Track.load_pseudo_mileage_dict>`.
        :type pseudo_mil_dict: dict | None
        :param ret_dat: Whether to return the processed data; defaults to ``False``.
        :type ret_dat: bool
        :return: INM combined data report with pseudo mileages.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft import FeatureCollator
            >>> fc = FeatureCollator(elr=['ECM7', 'ECM8'])
            Initialising the feature collator ... Done.
            >>> fc.assign_pseudo_mileage_to_inm_cdr()
            >>> fc.inm_combined_data_report.head()
                        RTE_NAME RTE_ORG_CODE  ... RELAYINGPOLICY TRACK_POSITION
            0  London North East           QG  ...              0            5.0
            1  London North East           QG  ...              0            7.0
            2  London North East           QG  ...              0            6.0
            3  London North East           QG  ...              0            4.0
            4  London North East           QG  ...              0            3.0
            [5 rows x 55 columns]
            >>> fc.inm_combined_data_report_.head()
                        RTE_NAME  ...                                    pseudo_geometry
            0  London North East  ...  LINESTRING Z (424635.8603999997 563764.4418000...
            1  London North East  ...  LINESTRING Z (424639.1655999999 563754.7467999...
            2  London North East  ...  LINESTRING Z (424637.4572000001 563759.7326999...
            3  London North East  ...  LINESTRING Z (424634.2836999996 563783.1506999...
            4  London North East  ...  LINESTRING Z (424633.2932000002 563788.2982999...
            [5 rows x 58 columns]

        **Illustration**::

            import numpy as np
            import matplotlib.pyplot as plt
            from pyhelpers.settings import mpl_preferences

            mpl_preferences(backend='TkAgg', font_name='Times New Roman')

            fig = plt.figure(figsize=(7, 7), constrained_layout=True)
            ax = fig.add_subplot(aspect='equal', adjustable='box')

            for g in fc.inm_combined_data_report_['pseudo_geometry']:
                g_ = np.array(g.coords)
                ax.plot(g_[:, 0], g_[:, 1])

            ax.set_xlabel('Easting', fontsize=14, labelpad=8.0)
            ax.set_ylabel('Northing', fontsize=14, labelpad=8.0)

            # from src.utils import cd_docs_source
            # from pyhelpers.store import save_figure
            # path_to_fig = cd_docs_source("_images\\fc_assign_pseudo_mileage_to_inm_cdr_demo")
            # save_figure(fig, path_to_fig + ".svg", verbose=True)
            # save_figure(fig, path_to_fig + ".pdf", verbose=True)

        .. figure:: ../_images/fc_assign_pseudo_mileage_to_inm_cdr_demo.*
            :name: fc_assign_pseudo_mileage_to_inm_cdr_demo
            :align: center
            :width: 100%

            Pseudo mileages for ECM8 INM combined data report.
        """

        inm_cdr_ = self.inm.load_combined_data_report(elr=self.elr if elr is None else elr)
        self.inm_combined_data_report = inm_cdr_

        inm_cdr_temp = [
            self.assign_pseudo_mileage(
                elr=elr_, data=dat, tid_col='REF_TRACKID',
                start_mil_col='ELR_STARTMEASURE', end_mil_col='ELR_ENDMEASURE',
                pseudo_mil_dict=pseudo_mil_dict,
            )
            for elr_, dat in list(inm_cdr_.groupby('ELR'))]

        self.inm_combined_data_report_ = pd.concat(inm_cdr_temp, axis=0).sort_index()

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # for g in inm_cdr.pseudo_geometry:
        #     g_arr = np.array(g)
        #     ax.plot(g_arr[:, 0], g_arr[:, 1])

        if ret_dat:
            return self.inm_combined_data_report_

    # == Geology ===================================================================================
    def _assign_pseudo_mileage_to_geology(self, elr=None, pseudo_mil_dict=None, ret_dat=False):
        """
        Assign pseudo mileage to the data of geology.

        :param elr: Engineer's Line Reference(s); defaults to ``None``.
        :type elr: str | list | None
        :param pseudo_mil_dict: Data of pseudo mileages;
            when ``pseudo_mil_dict=None`` (default), it retrieves the data by using the method
            :meth:`Track.load_pseudo_mileage_dict()
            <src.preprocessor.Track.load_pseudo_mileage_dict>`.
        :type pseudo_mil_dict: dict | None
        :param ret_dat: Whether to return the processed data; defaults to ``False``.
        :type ret_dat: bool
        :return: Data of geological features with assigned pseudo mileages.
        :rtype: list

        **Examples**::

            >>> from src.shaft import FeatureCollator
            >>> fc = FeatureCollator(elr=['ECM7', 'ECM8'])
            Initialising the feature collator ... Done.
            >>> fc._assign_pseudo_mileage_to_geology()
            >>> # fc.geological_features
            >>> # fc.geological_features_
        """

        self.geological_features = self.geology.load_summary(elr=self.elr if elr is None else elr)
        # geology.loc[geology['Start'] < 0, 'Start'] = 0

        if pseudo_mil_dict is None:
            pseudo_trk_mileage_dict = self.track.load_pseudo_mileage_dict(elr=elr)
        else:
            pseudo_trk_mileage_dict = pseudo_mil_dict.copy()

        pseudo_track_mileage_dict = copy.copy(pseudo_trk_mileage_dict)
        for k, v in pseudo_trk_mileage_dict.items():
            pseudo_track_mileage_dict[k] = pd.concat(v.values(), axis=0)

        geol_dat_list = []
        for geol_elr, geol_dat in self.geological_features.groupby('ELR'):
            pseudo_trk_mil = pseudo_track_mileage_dict[geol_elr]
            geol_dat_ = geol_dat.join(pseudo_trk_mil, on='Start')
            geol_dat_ = geol_dat_.join(pseudo_trk_mil, on='End', lsuffix='_start', rsuffix='_end')

            geol_dat_list.append(geol_dat_)

        self.geological_features_ = geol_dat_list

        if ret_dat:
            return self.geological_features_

    # == Track quality =============================================================================
    def calculate_track_quality_stats(self, elr=None, tq_date=None, agg_method=None):
        """
        Calculate basic statistics of track quality data.

        :param elr: Engineer's Line Reference(s); defaults to ``None``.
        :type elr: str | list | None
        :param tq_date: Dates of track quality data; defaults to ``None``.
        :type tq_date: list | None
        :param agg_method: Method of aggregation;
            when ``agg_method=None`` (default), it calculates the average of grouped data;
            an alternative method is 'rms', i.e. 'root-mean-square'.
        :return: Basic statistics of track quality data.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft import FeatureCollator
            >>> fc = FeatureCollator(elr='ECM8')
            Initialising the feature collator ... Done.
            >>> tq_stats_data = fc.calculate_track_quality_stats()
            >>> tq_stats_data
                   ELR  Track Id  ...  Dipped Left (mrad)  Dipped Right (mrad)
            0     ECM8      1100  ...                 0.0               0.0000
            1     ECM8      1100  ...                 0.0               0.0000
            2     ECM8      1100  ...                 0.0               0.0000
            3     ECM8      1100  ...                 0.0              -0.4313
            4     ECM8      1100  ...                 0.0              -1.8274
                ...       ...  ...                 ...                  ...
            7021  ECM8      2100  ...                 0.0               0.0000
            7022  ECM8      2100  ...                 0.0               0.0000
            7023  ECM8      2100  ...                 0.0               0.0000
            7024  ECM8      2100  ...                 0.0               0.0000
            7025  ECM8      2100  ...                 0.0               0.0000
            [7026 rows x 18 columns]
        """

        if tq_date is None:
            tq_dates = ['2019-10-15', '2020-04-15']
        else:
            tq_dates = tq_date.copy()

        elr_ = self.elr if elr is None else elr
        self.track_quality = self.track.load_track_quality(elr=elr_, tq_date=tq_dates)

        stats_group_columns = ['ELR', 'Track Id', 'Mileage']

        tq_stats_list = []

        time_delta = datetime.timedelta(days=30)

        for elr_tid, dat in self.track_quality.groupby(['ELR', 'Track Id']):
            tq_dates_actual = dat.Date.unique()
            tq_dates_label = dict(zip(tq_dates_actual, [''] * len(tq_dates_actual)))

            for d in tq_dates:
                d_ = dateutil.parser.parse(d)
                for d_actual in tq_dates_actual:
                    d_actual_ = dateutil.parser.parse(d_actual)
                    if d_ - time_delta <= d_actual_ <= d_ + time_delta:
                        tq_month_label = '0' + str(d_.month) if d_.month < 10 else str(d_.month)
                        tq_dates_label[d_actual] = str(d_.year) + tq_month_label

            dat['PseudoDate'] = dat['Date'].replace(tq_dates_label)

            (p_date_earlier, tq_earlier), (p_date_later, tq_later) = dat.groupby('PseudoDate')
            loc_columns = [
                'ELR', 'Track Id', 'Start Eighth', 'End Eighth',
                'Locn (mile)', 'Locn (chs)', 'Locn (yards)']
            tq_earlier_ = tq_earlier.drop_duplicates(loc_columns, keep='last')
            tq_later_ = tq_later.drop_duplicates(loc_columns, keep='last').set_index(loc_columns)

            suffices = [f'_{p_date_earlier}', f'_{p_date_later}']
            tq_merged = tq_earlier_.join(tq_later_, on=loc_columns, rsuffix=suffices[1])

            variable_names = [x for x in dat.columns if x not in loc_columns and 'Date' not in x]

            tq_earlier_dat = tq_merged[variable_names]
            tq_later_dat = tq_merged[[x + suffices[1] for x in variable_names]]
            tq_later_dat.columns = variable_names

            if agg_method == 'rms':
                var_val = (tq_later_dat - tq_earlier_dat) ** 2
                stats_func_ = [lambda x: np.sqrt(np.sum(x) / len(x))]
            else:
                var_val = tq_later_dat - tq_earlier_dat
                stats_func_ = [np.nanmean]
            stats_calc = dict(zip(variable_names, stats_func_ * len(variable_names)))

            tq_valid = tq_merged[loc_columns + ['Date', 'Date' + suffices[1]]].join(var_val)
            tq_valid['Mileage'] = tq_valid['Locn (mile)'] + 10 ** -4 * tq_valid['Locn (chs)'] * 22

            tq_valid_stats = tq_valid.groupby(stats_group_columns).aggregate(stats_calc)
            tq_valid_stats.reset_index(inplace=True)

            tq_stats_list.append(tq_valid_stats)

        track_quality_stats = pd.concat(tq_stats_list, axis=0).sort_values(
            stats_group_columns, ignore_index=True)

        return track_quality_stats

    def assign_pseudo_mileage_to_track_quality(self, elr=None, tq_date=None, agg_method=None,
                                               pseudo_mil_dict=None, ret_dat=False):
        """
        Assign pseudo mileage to the preprocessed track quality data.

        :param elr: Engineer's Line Reference(s); defaults to ``None``.
        :type elr: str | list | None
        :param tq_date: Dates of track quality data; defaults to ``None``.
        :type tq_date: list | None
        :param agg_method: Method of aggregation;
            when ``agg_method=None`` (default), it calculates the average of grouped data;
            an alternative method is 'rms', i.e. 'root-mean-square'.
        :param pseudo_mil_dict: Data of pseudo mileages;
            when ``pseudo_mil_dict=None`` (default), it retrieves the data by using the method
            :meth:`Track.load_pseudo_mileage_dict()
            <src.preprocessor.Track.load_pseudo_mileage_dict>`.
        :type pseudo_mil_dict: dict | None
        :param ret_dat: Whether to return the processed data; defaults to ``False``.
        :type ret_dat: bool
        :return: Track quality data with assigned pseudo mileages.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft import FeatureCollator
            >>> fc = FeatureCollator(elr='ECM8')
            Initialising the feature collator ... Done.
            >>> fc.assign_pseudo_mileage_to_track_quality()
            >>> fc.track_quality
                      ELR  Track Id  ... Dipped Left (mrad) Dipped Right (mrad)
            0        ECM8      2100  ...                0.0                 0.0
            1        ECM8      2100  ...                0.0                 0.0
            2        ECM8      1100  ...                0.0                 0.0
            3        ECM8      1100  ...                0.0                 0.0
            4        ECM8      2100  ...                0.0                 0.0
                   ...       ...  ...                ...                 ...
            1375425  ECM8      1100  ...                0.0                 0.0
            1375426  ECM8      1100  ...                0.0                 0.0
            1375427  ECM8      1100  ...                0.0                 0.0
            1375428  ECM8      1100  ...                0.0                 0.0
            1375429  ECM8      1100  ...                0.0                 0.0
            [1375430 rows x 23 columns]
            >>> fc.track_quality_
                   ELR  ...                                  pseudo_geometry
            0     ECM8  ...  POINT Z (340000.8967833333 674040.6199604174 0)
            1     ECM8  ...    POINT Z (340019.6609500001 674047.63910625 0)
            2     ECM8  ...  POINT Z (340038.4317759821 674054.6556251637 0)
            3     ECM8  ...  POINT Z (340057.3424475812 674061.6169787806 0)
            4     ECM8  ...  POINT Z (340076.2531191803 674068.5783323975 0)
                ...  ...                                              ...
            7021  ECM8  ...  POINT Z (397758.8943053244 657528.7732021383 0)
            7022  ECM8  ...  POINT Z (397770.3546583527 657512.2525615764 0)
            7023  ECM8  ...  POINT Z (397781.5221755836 657495.5327715443 0)
            7024  ECM8  ...  POINT Z (397792.4366677299 657478.6479913326 0)
            7025  ECM8  ...  POINT Z (397803.1121100634 657461.6104920406 0)
            [7026 rows x 19 columns]
        """

        track_quality_stats = self.calculate_track_quality_stats(
            elr=elr, tq_date=tq_date, agg_method=agg_method)

        if pseudo_mil_dict is None:
            pseudo_trk_shp_mileage_dict = self.track.load_pseudo_mileage_dict(elr=elr)
        else:
            pseudo_trk_shp_mileage_dict = copy.copy(pseudo_mil_dict)

        tq_stats = []
        for (elr_, tid), dat in list(track_quality_stats.groupby(['ELR', 'Track Id'])):
            pseudo_mileage_dat = pseudo_trk_shp_mileage_dict[elr_][tid]
            dat_ = dat.join(pseudo_mileage_dat, on='Mileage')
            tq_stats.append(dat_)

        self.track_quality_ = pd.concat(tq_stats, axis=0).sort_index()

        if ret_dat:
            return self.track_quality_

    # == Integration of the feature data ===========================================================
    @staticmethod
    def _get_structure_data(structure_cls, **kwargs):
        structure_func = [
            structure_cls.__getattribute__(x) for x in dir(structure_cls)
            if x.startswith('load') and x.endswith('_shp')]

        structure_names = [
            structure_cls.__getattribute__(x) for x in dir(structure_cls)
            if x.endswith('_DIRNAME') and not x.startswith(('PROJ', 'STRUCT'))]

        structure_data = {
            name: func(**kwargs) for func, name in zip(structure_func, structure_names)}

        return structure_data

    def integrate_data(self, element, direction, subsect_len, elr=None, ret_dat=False,
                       verbose=False, **kwargs):
        """
        Construct an integrated data set for the development of a machine learning model
        for track fixity.

        :param element: Element of rail head, such as left or right top of rail or running edge.
        :type element: str
        :param direction: Railway direction, such as up or down direction.
        :type direction: str
        :param subsect_len: Length (in metre) of a subsection for which movement are calculated;
            defaults to ``10``.
        :type subsect_len: int
        :param elr: Engineer's Line Reference(s); defaults to ``None``.
        :type elr: str | list | None
        :param ret_dat: Whether to return the created data set; defaults to ``False``.
        :type ret_dat: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: An integrated data set for the development of a machine learning model
            for track fixity.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft import FeatureCollator
            >>> fc = FeatureCollator(elr=['ECM7', 'ECM8'])
            Initialising the feature collator ... Done.
            >>> fc.integrate_data(
            ...     element='Left Top', direction='Up', subsect_len=10, verbose=True)
            >>> fc.track_movement_
                                                         subsection  ...  Stations
            0     LINESTRING Z (399428.96 653473.9 34.442, 39942...  ...         0
            1     LINESTRING Z (399434.5201166851 653482.2116640...  ...         0
            2     LINESTRING Z (399440.0608295194 653490.5362699...  ...         0
            3     LINESTRING Z (399445.5700898784 653498.8817273...  ...         0
            4     LINESTRING Z (399451.0456262118 653507.2493444...  ...         0
                                                             ...  ...       ...
            7625  LINESTRING Z (340180.451306418 674103.12674018...  ...         0
            7626  LINESTRING Z (340171.0809418422 674099.6346047...  ...         0
            7627  LINESTRING Z (340161.7096778383 674096.1448800...  ...         0
            7628  LINESTRING Z (340152.3387371677 674092.6542742...  ...         0
            7629  LINESTRING Z (340142.9690238989 674089.1603813...  ...         0
            [7630 rows x 46 columns]
            >>> fc.track_movement_.columns.to_list()[-10:]
            ['vertical_displacement_b_abs_max',
             'Curvature',
             'Cant',
             'Max speed',
             'Max axle load',
             'Overline bridges',
             'Retaining walls',
             'Tunnels',
             'Underline bridges',
             'Stations']
        """

        elr_ = self.elr if elr is None else copy.copy(elr)

        if verbose:
            print("Calculating track movement", end=" ... ")

        try:
            self.track_movement = self.load_movement(
                element=element, direction=direction, subsect_len=subsect_len, **kwargs)

            self.element = element
            self.direction = direction
            self.subsect_len = subsect_len

            # track_movement = fc.track_movement['Up_LeftTopOfRail_201910_202004']
            track_movement = list(self.track_movement.values())[0]

            if verbose:
                print("Done.")

        except Exception as e:
            _print_failure_msg(e)

            return None

        if verbose:
            print("Collating features: ")

        feature_data = []

        # -- Ballast -------------------------------------------------------------------------------
        if verbose:
            print("  ballasts (src. Ballast summary)", end=" ... ")

        try:
            self.assign_pseudo_mileage_to_ballast(
                elr=elr_, pseudo_mil_dict=self.pseudo_mileage_dict)
            self.collate_ballast_features()

            if verbose:
                print("Done.")

        except Exception as e:
            _print_failure_msg(e)

        feature_data.append(self.ballast_features)

        # -- CARRS and OPAS ------------------------------------------------------------------------
        if verbose:
            print("  structures (src. CARRS and OPAS)", end=" ... ")

        try:
            carrs_dict = self._get_structure_data(self.carrs, elr=elr_)
            opas_dict = self._get_structure_data(self.opas)

            structure_data_dict = {**carrs_dict, **opas_dict}
            for structure_name, structure_data in structure_data_dict.items():
                self.collate_nearest_structure(
                    structure_data=structure_data, structure_name=structure_name)

            if verbose:
                print("Done.")

        except Exception as e:
            _print_failure_msg(e)

        feature_data += list(self.structures_.values())

        # -- Geology -------------------------------------------------------------------------------
        self._assign_pseudo_mileage_to_geology(elr=elr_, pseudo_mil_dict=self.pseudo_mileage_dict)

        # -- INM -----------------------------------------------------------------------------------
        self.assign_pseudo_mileage_to_inm_cdr(elr=elr_, pseudo_mil_dict=self.pseudo_mileage_dict)

        # == Integration ===========================================================================
        if verbose:
            print("Finalising the data integration", end=" ... ")

        try:
            feature_data_ = pd.concat([x.reset_index(drop=True) for x in feature_data], axis=1)
            self.track_movement_ = track_movement.join(feature_data_)

            if verbose:
                print("Done.")

        except Exception as e:
            _print_failure_msg(e)

        if ret_dat:
            return self.track_movement_
