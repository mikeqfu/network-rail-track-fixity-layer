"""
A class for preprocessing data of spreadsheet-based *reports*.
"""

import glob
import os
import re

import numpy as np
import pandas as pd
from pyhelpers.dirs import cd, cdd
from pyhelpers.store import load_data, save_data
from pyrcs.converter import fix_mileage

from src.preprocessor import INM
from src.utils.general import TrackFixityDB


class Reports:
    """
    Spreadsheet-based reports.

    .. note::

        The data handled by this class included a number of spreadsheet reports up to 2020-08-22,
        regarding the rail tracks. These data were not yet explored for this project.
    """

    #: Data name.
    NAME: str = 'Reports'
    #: Pathname of a local directory where the data is stored.
    DATA_DIR: str = os.path.relpath(cdd("Reports", "20200822"))

    #: Filename of the data of track report.
    TRK_REPORT_FILENAME: str = "9999_Track_report.zip"
    #: Filename of the data of defrag report.
    DEFRAG_REPORT_FILENAME: str = "Defrag_report.zip"
    #: Filename of the data of equated track miles.
    ETM_FILENAME: str = "Equated_track_miles.zip"
    #: Filename of the data of INM combined data report.
    INM_CDR_FILENAME: str = "INM_combined_data_report.zip"
    #: Filename of the data of INM combined report for DST.
    INM_CDR_DST_FILENAME: str = "INM_combined_report_for_DST.zip"
    #: Filename of the data of junctions.
    JCT_FILENAME: str = "Junctions.zip"
    #: Filename of the data of plain line SnC attributes gap.
    PLSAQ_FILENAME: str = "Plain_line_SnC_attributes_gap.zip"
    #: Filename of the data of plain line SnC attributes overlap.
    PLSAO_FILENAME: str = "Plain_line_SnC_attributes_overlap.zip"
    #: Filename of the data of responsibility gap report.
    RGR_FILENAME: str = "Responsibility_gap_report.zip"
    #: Filename of the data of track attributes gap report.
    TAGR_FILENAME: str = "Track_attributes_gap_report.zip"
    #: Filename of the data of track category gap report.
    TCGR_FILENAME: str = "Track_category_gap_report.zip"
    #: Filename of the data of track category with responsibility.
    TCWR_FILENAME: str = "Track_category_with_responsibility.zip"
    #: Filename of the data of track category with responsibility and switch.
    TCWRS_FILENAME: str = "Track_category_with_responsibility_and_switch.zip"

    def __init__(self, db_instance=None):
        """
        :param db_instance: PostgreSQL database instance; defaults to ``None``.
        :type db_instance: TrackFixityDB | None

        :ivar list data_filenames: Filenames of all the original data files.
        :ivar list data_pathnames: Pathnames of all the original data files.
        :ivar dict mileage_dtypes: Data type of mileages.
        :ivar TrackFixityDB db_instance: PostgreSQL database instance.

        **Examples**::

            >>> from src.preprocessor import Reports
            >>> reports = Reports()
            >>> reports.NAME
            'Reports'
        """

        self.data_filenames = glob.glob1(cd(self.DATA_DIR), "*.zip")
        self.data_pathnames = glob.glob(cd(self.DATA_DIR, "*.zip"))

        self.mileage_dtypes = {'ELR_STARTMEASURE': str, 'ELR_ENDMEASURE': str}

        self.db_instance = db_instance

    def cdd(self, *filename, mkdir=True):
        """
        Change directory to the report pack.

        :param filename: Name of a file.
        :type filename: str
        :param mkdir: Whether to create a subdirectory (if it does not exist);
            defaults to ``True``.
        :type mkdir: bool
        :return: Pathname of the local directory of the report pack
            (and subdirectories and/or files within it).
        :rtype: str
        """

        path = cd(self.DATA_DIR, *filename, mkdir=mkdir)

        return path

    def read_inm_cdr(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read INM combined report data.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: INM combined report data.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import Reports
            >>> reports = Reports()
            >>> # inm_cdr = reports.read_inm_cdr(update=True, verbose=True)
            >>> inm_cdr = reports.read_inm_cdr()
            >>> inm_cdr.shape
            (487187, 55)

        .. note::

            - ``'SLEEPERSPER60FOOTLENGTH'`` (int): strings (i.e. 'PAN' and 'NONE') and ``nan``
            - ``'PATCHPROGRAM'`` (str of 2 digits of int): 2019, 'Patch Program' and ``nan``
            - ``'TRACK_CATEGORY'`` (int, 1-6): ``nan`` and '1A'
            - ``'SMOOTH_TRACK_CAT'`` (int, 1-6): ``nan`` and '1A'
        """

        path_to_csv = self.cdd(self.INM_CDR_FILENAME)

        path_to_pkl = re.sub(r"\.zip$|\.pickle$", ".pkl", path_to_csv)

        if os.path.isfile(path_to_pkl) and not update:
            inm_cdr = load_data(path_to_pkl)

        else:
            inm = INM()

            inm_cdr = pd.read_csv(path_to_csv, dtype=inm.cdr_dtypes)

            mileage_col = list(inm.mileage_dtypes.keys())
            inm_cdr[mileage_col] = inm_cdr[mileage_col].map(lambda x: fix_mileage(x.strip()))

            str_col = list(inm.cdr_dtypes_object.keys())
            inm_cdr.replace(dict(zip(str_col, [{np.nan: ''}] * len(str_col))), inplace=True)

            save_data(inm_cdr, path_to_pkl, verbose=verbose)

        return inm_cdr

    def read_inm_dst(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read INM combined report data for DST (Decision Support Tool).

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: INM combined report data for DST.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import Reports
            >>> reports = Reports()
            >>> # inm_dst = reports.read_inm_dst(update=True, verbose=True)
            >>> inm_dst = reports.read_inm_dst()
            >>> inm_dst.shape
            (489973, 60)

        .. note::

            - ``'SLEEPERSPER60FOOTLENGTH'`` (int): nan, 'N', 'PAN', 'NONE' and
              'AoR Rec without value'
            - ``'PATCHPROGRAM'`` (str of 2 digits of int): nan, 2019, 'Patch Program' and
              'AoR Rec without value'
            - ``'TRACK_CATEGORY'`` (int, 1-6): nan, '1A' and
              'Track Category without value'
            - ``'SMOOTH_TRACK_CAT'`` (int, 1-6): nan and '1A'
            - ``'SPEED_LEVEL_RAISED'``: nan and 'Track Category without value'
        """

        path_to_csv = self.cdd(self.INM_CDR_DST_FILENAME)

        path_to_pkl = re.sub(r"\.zip$|\.pickle$", ".pkl", path_to_csv)

        if os.path.isfile(path_to_pkl) and not update:
            inm_dst = load_data(path_to_pkl)

        else:
            err_cols = ['SLEEPERSPER60FOOTLENGTH',
                        'PATCHPROGRAM',
                        'TRACK_CATEGORY',
                        'SMOOTH_TRACK_CAT',
                        'SPEED_LEVEL_RAISED']
            mil_cols = list(self.mileage_dtypes.keys())

            col_type = dict(zip(mil_cols + err_cols, len(mil_cols + err_cols) * [str]))
            inm_dst = pd.read_csv(path_to_csv, dtype=col_type)

            # data.SLEEPERSPER60FOOTLENGTH.unique()

            inm_dst[mil_cols] = inm_dst[mil_cols].map(lambda x: x.strip())

            save_data(inm_dst, path_to_pkl, verbose=verbose)

        return inm_dst

    def read_track_report(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read data of track report.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of track report.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import Reports
            >>> reports = Reports()
            >>> # track_report = reports.read_track_report(update=True, verbose=True)
            >>> track_report = reports.read_track_report()
            >>> track_report.shape
            (325, 13)
        """

        path_to_csv = self.cdd(self.TRK_REPORT_FILENAME)

        path_to_pkl = re.sub(r"\.zip$|\.pickle$", ".pkl", path_to_csv)

        if os.path.isfile(path_to_pkl) and not update:
            track_report = load_data(path_to_pkl)

        else:
            track_report = pd.read_csv(path_to_csv, dtype=self.mileage_dtypes)

            track_report.replace({np.nan: None}, inplace=True)

            save_data(track_report, path_to_pkl, verbose=verbose)

        return track_report

    def read_defrag_report(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read data of defrag report.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of defrag report.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import Reports
            >>> reports = Reports()
            >>> # defrag_report = reports.read_defrag_report(update=True, verbose=True)
            >>> defrag_report = reports.read_defrag_report()
            >>> defrag_report.shape
            (529176, 71)

        .. note::

            - ``'REF_TRACKID'`` (``int``): ``0`` and ``nan``
            - ``'SMOOTH_TRACK_CAT'`` (``int``, 1-6): ``nan`` and ``'1A'``
            - ``'TRACK_CATEGORY'`` (``int``, 1-6): ``nan``,
              ``'Track Category without value'`` and '1A'
            - ``'SPEED_LEVEL_RAISED'``: ``nan``, ``'Track Category without value'``
        """

        path_to_csv = self.cdd(self.DEFRAG_REPORT_FILENAME)

        path_to_pkl = re.sub(r"\.zip$|\.pickle$", ".pkl", path_to_csv)

        if os.path.isfile(path_to_pkl) and not update:
            defrag_report = load_data(path_to_pkl)

        else:
            err_cols = ['REF_TRACKID',
                        'SMOOTH_TRACK_CAT',
                        'TRACK_CATEGORY',
                        'SPEED_LEVEL_RAISED']
            col_type = dict(zip(err_cols, len(err_cols) * [str]))

            col_type.update(self.mileage_dtypes)
            defrag_report = pd.read_csv(path_to_csv, dtype=col_type)

            defrag_report.replace({np.nan: None}, inplace=True)

            save_data(defrag_report, path_to_pkl, verbose=verbose)

        return defrag_report

    def read_etm(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read data of equated track miles.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of equated track miles.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import Reports
            >>> reports = Reports()
            >>> # equated_track_miles = reports.read_etm(update=True, verbose=True)
            >>> equated_track_miles = reports.read_etm()
            >>> equated_track_miles.shape
            (37953, 8)
        """

        path_to_csv = self.cdd(self.ETM_FILENAME)

        path_to_pkl = re.sub(r"\.zip$|\.pickle$", ".pkl", path_to_csv)

        if os.path.isfile(path_to_pkl) and not update:
            equated_track_miles = load_data(path_to_pkl)

        else:
            equated_track_miles = pd.read_csv(path_to_csv, dtype=self.mileage_dtypes)

            mileage_cols = list(self.mileage_dtypes.keys())
            equated_track_miles[mileage_cols] = equated_track_miles[mileage_cols].map(fix_mileage)

            save_data(equated_track_miles, path_to_pkl, verbose=verbose)

        return equated_track_miles

    def read_junctions(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read data of junctions.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of junctions.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import Reports
            >>> reports = Reports()
            >>> # junctions = reports.read_junctions(update=True, verbose=True)
            >>> junctions = reports.read_junctions()
            >>> junctions.shape
            (3990, 8)

        .. note::

            from pyrcs.utils import nr_mileage_to_yards

            end_m = junctions_csv.ELR_ENDMEASURE[0]
            start_m = junctions_csv.ELR_STARTMEASURE[0]
            total_m_j = junctions_csv.TOTALMILEGE[0]
            total_m_sandc = junctions_csv.TOTALMILEGE[1]

            total_yards = nr_mileage_to_yards(end_m) - nr_mileage_to_yards(start_m)

            total_mi = total_m_j + total_m_sandc
            total_m_yards = round(measurement.measures.Distance(mi=total_mi).yd)

            total_yards == total_m_yards  # True
        """

        path_to_csv = self.cdd(self.JCT_FILENAME)

        path_to_pkl = re.sub(r"\.zip$|\.pickle$", ".pkl", path_to_csv)

        # noinspection SpellCheckingInspection
        if os.path.isfile(path_to_pkl) and not update:
            junctions = load_data(path_to_pkl)

        else:
            # noinspection SpellCheckingInspection
            col_type = {'TOTALMILEGE': np.float64, 'EQA Milege': np.float64}
            # noinspection PyTypeChecker
            col_type.update(self.mileage_dtypes)

            junctions = pd.read_csv(path_to_csv, dtype=col_type)

            # junctions_csv.rename(columns={'EQA Milege': 'EQA Mileage'}, inplace=True)

            save_data(junctions, path_to_pkl, verbose=verbose)

        return junctions

    def read_plsag(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read data of plain line SnC attributes gap.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of plain line SnC attributes gap.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import Reports
            >>> reports = Reports()
            >>> # plsag = reports.read_plsag(update=True, verbose=True)
            >>> plsag = reports.read_plsag()
            >>> plsag.shape
            (222804, 19)
        """

        path_to_csv = self.cdd(self.PLSAQ_FILENAME)

        path_to_pkl = re.sub(r"\.zip$|\.pickle$", ".pkl", path_to_csv)

        if os.path.isfile(path_to_pkl) and not update:
            plsag = load_data(path_to_pkl)

        else:
            mil_cols = ['GAP_START_MIL', 'GAP_FINISH_MIL', 'GAP_LENGTH',
                        'Track/SandC Start Mil', 'Track/SandC End Mil',
                        'Nxt Track/SandC Start Mil', 'Nxt Track/SandC End Mil']
            col_type = dict(zip(mil_cols, [np.float64] * len(mil_cols)))

            plsag = pd.read_csv(path_to_csv, dtype=col_type)

            plsag.replace({np.nan: None}, inplace=True)

            save_data(plsag, path_to_pkl, verbose=verbose)

        return plsag

    def read_plsao(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read data of plain line SnC attributes overlap.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of plain line SnC attributes overlap.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import Reports
            >>> reports = Reports()
            >>> # plsao = reports.read_plsao(update=True, verbose=True)
            >>> plsao = reports.read_plsao()
            >>> plsao.shape
            (163725, 13)
        """

        path_to_csv = self.cdd(
            self.PLSAO_FILENAME)

        path_to_pkl = re.sub(r"\.zip$|\.pickle$", ".pkl", path_to_csv)

        if os.path.isfile(path_to_pkl) and not update:
            plsao = load_data(path_to_pkl)

        else:
            mil_cols = ['OVERLAP_START_MIL', 'OVERLAP_END_MIL', 'OVERLAP_LENGTH',
                        'SANDC_START_MIL', 'SANDC_END_MIL',
                        'TRACK_START_MIL', 'TRACK_END_MIL']
            col_type = dict(zip(mil_cols, [np.float64] * len(mil_cols)))

            plsao = pd.read_csv(path_to_csv, dtype=col_type)

            plsao.replace({np.nan: None}, inplace=True)

            save_data(plsao, path_to_pkl, verbose=verbose)

        return plsao

    def read_rgr(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read report data of responsibility gap.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of responsibility gap.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import Reports
            >>> reports = Reports()
            >>> # responsibility_gap = reports.read_rgr(update=True, verbose=True)
            >>> responsibility_gap = reports.read_rgr()
            >>> responsibility_gap.shape
            (56170, 14)
        """

        path_to_csv = self.cdd(self.RGR_FILENAME)

        path_to_pkl = re.sub(r"\.zip$|\.pickle$", ".pkl", path_to_csv)

        if os.path.isfile(path_to_pkl) and not update:
            responsibility_gap = load_data(path_to_pkl)

        else:
            responsibility_gap = pd.read_csv(path_to_csv, dtype=self.mileage_dtypes)

            responsibility_gap.replace({np.nan: None}, inplace=True)

            save_data(responsibility_gap, path_to_pkl, verbose=verbose)

        return responsibility_gap

    def read_tagr(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read report data of track attributes gap.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of track attributes gap.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import Reports
            >>> reports = Reports()
            >>> # tagr = reports.read_tagr(update=True, verbose=True)
            >>> tagr = reports.read_tagr()
            >>> tagr.shape
            (303327, 38)
        """

        path_to_csv = self.cdd(self.TAGR_FILENAME)

        path_to_pkl = re.sub(r"\.zip$|\.pickle$", ".pkl", path_to_csv)

        if os.path.isfile(path_to_pkl) and not update:
            tagr = load_data(path_to_pkl)

        else:
            tagr = pd.read_csv(path_to_csv, dtype=self.mileage_dtypes)

            mil_cols = list(self.mileage_dtypes.keys())
            tagr[mil_cols] = tagr[mil_cols].map(lambda x: fix_mileage(x.strip()))

            tagr.replace({np.nan: None}, inplace=True)

            save_data(tagr, path_to_pkl, verbose=verbose)

        return tagr

    def read_tcgr(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read report data of track category gap report.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of track category gap report.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import Reports
            >>> reports = Reports()
            >>> # tcgr = reports.read_tcgr(update=True, verbose=True)
            >>> tcgr = reports.read_tcgr()
            >>> tcgr.shape
            (253202, 16)
        """

        path_to_csv = self.cdd(self.TCGR_FILENAME)

        path_to_pkl = re.sub(r"\.zip$|\.pickle$", ".pkl", path_to_csv)

        if os.path.isfile(path_to_pkl) and not update:
            tcgr = load_data(path_to_pkl)

        else:
            tcgr = pd.read_csv(path_to_csv, dtype=self.mileage_dtypes)

            mil_cols = list(self.mileage_dtypes.keys())
            tcgr[mil_cols] = tcgr[mil_cols].map(lambda x: fix_mileage(x.strip()))

            tcgr.replace({'OVERRIDE': {np.nan: None}}, inplace=True)

            save_data(tcgr, path_to_pkl, verbose=verbose)

        return tcgr

    def read_tcr(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read data of track category with responsibility.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of track category with responsibility.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import Reports
            >>> reports = Reports()
            >>> # tcr_ = reports.read_tcr(update=True, verbose=True)
            >>> tcr_ = reports.read_tcr()
            >>> tcr_.shape
            (251512, 24)
        """

        path_to_csv = self.cdd(self.TCWR_FILENAME)

        path_to_pkl = re.sub(r"\.zip$|\.pickle$", ".pkl", path_to_csv)

        if os.path.isfile(path_to_pkl) and not update:
            tcr = load_data(path_to_pkl)

        else:
            tcr = pd.read_csv(path_to_csv, dtype=self.mileage_dtypes)

            mil_cols = list(self.mileage_dtypes.keys())
            tcr[mil_cols] = tcr[mil_cols].map(lambda x: fix_mileage(x.strip()))

            str_cols = [x for x in tcr.columns if tcr[x].dtype == 'O']
            tcr.replace(dict(zip(str_cols, [{np.nan: None}] * len(str_cols))), inplace=True)

            save_data(tcr, path_to_pkl, verbose=verbose)

        return tcr

    def read_tcrs(self, update=False, verbose=False):
        # noinspection PyShadowingNames
        """
        Read data of track category with responsibility and switch.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of track category with responsibility and switch.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import Reports
            >>> reports = Reports()
            >>> # tcrs = reports.read_tcrs(update=True, verbose=True)
            >>> tcrs = reports.read_tcrs()
            >>> tcrs.shape
            (320799, 21)

        .. note::

            ``'UNIQUE_ID'`` (``int``): ``nan`` and ``'NOTINGEOGIS'``
        """

        path_to_csv = self.cdd(self.TCWRS_FILENAME)

        path_to_pkl = re.sub(r"\.zip$|\.pickle$", ".pkl", path_to_csv)

        if os.path.isfile(path_to_pkl) and not update:
            tcrs = load_data(path_to_pkl)

        else:
            col_type = {'UNIQUE_ID': str}
            col_type.update(self.mileage_dtypes)
            tcrs = pd.read_csv(path_to_csv, dtype=col_type)

            mil_cols = list(self.mileage_dtypes.keys())
            tcrs[mil_cols] = tcrs[mil_cols].map(lambda x: fix_mileage(x.strip()))

            str_cols = [x for x in tcrs.columns if tcrs[x].dtype == 'O']
            tcrs.replace(dict(zip(str_cols, [{np.nan: None}] * len(str_cols))), inplace=True)

            tcrs.UNIQUE_ID = tcrs.UNIQUE_ID.map(lambda x: x.strip('0') if x else x)

            save_data(tcrs, path_to_pkl, verbose=verbose)

        return tcrs


# if __name__ == '__main__':
#     reports = Reports()
#
#     for meth_name in dir(reports):
#         if meth_name.startswith('read_'):
#             meth = getattr(reports, meth_name)
#             meth(update=True, verbose=True)
