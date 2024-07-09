"""
A class for preprocessing data from *Ground Penetrating Radar (GPR)*.
"""

import collections
import copy
import csv
import datetime
import gc
import glob
import itertools
import os
import re
import shutil
import struct

import numpy as np
import pandas as pd
import siina
import xmltodict
from pyhelpers._cache import _print_failure_msg
from pyhelpers.dirs import cd, cdd
from pyhelpers.ops import confirmed, get_dict_values
from pyhelpers.store import load_data, save_data, unzip
from pyhelpers.text import get_acronym

from src.utils.general import TrackFixityDB


class GPR:
    """
    *Ground Penetrating Radar*.

    .. note::

        - This class handles GPR data, formatted as a bundle of three types of data files  with the
          extensions ".DZX", ".DZG", and ".DZT".
        - GPR data has not yet been considered in the :mod:`~src.shaft` module for developing the
          current data model for this project. However, it may potentially be useful and could be
          included for further development of the data model and the algorithm for learning about
          track fixity.
    """

    #: Data name.
    NAME: str = 'Ground Penetrating Radar'
    #: Acronym for the data name.
    ACRONYM: str = get_acronym(NAME)
    #: Pathname of a local directory where the GPR data is stored.
    DATA_DIR: str = os.path.relpath(cdd(ACRONYM))
    #: Pathname of a local directory where zipped GPR data is stored for backup.
    ZIPFILE_DATA_DIR: str = os.path.relpath(cdd("_backups", ACRONYM, "files_zipped"))
    #: Pathname of a local directory where the GPR data files are stored.
    FILE_DATA_DIR: str = os.path.relpath(cd(DATA_DIR, "files"))
    #: Name of the schema for storing the GPR data.
    SCHEMA_NAME: str = copy.copy(ACRONYM)
    #: Name of the table for storing *DZX* data.
    DZX_TABLE_NAME: str = 'DZX'
    #: Name of the table for storing *DZG* data.
    DZG_TABLE_NAME: str = 'DZG'
    #: Name of the table for storing *DZT* data.
    DZT_TABLE_NAME: str = 'DZT'

    def __init__(self, db_instance=None):
        """
        :param db_instance: PostgreSQL database instance; defaults to ``None``.
        :type db_instance: TrackFixityDB | None

        :ivar list data_dates: Date range of the original GPR data.
        :ivar dict dtypes: Field types of the GPR data.
        :ivar TrackFixityDB db_instance: PostgreSQL database instance used for operations.
        :ivar typing.Callable schema_name_: Name of the schema for storing the GPR data.

        **Examples**::

            >>> from src.preprocessor import GPR
            >>> gpr = GPR()
            >>> gpr.NAME
            'Ground Penetrating Radar'
            >>> gpr.data_dates  # '20200312 PM' -> '20200312'
            ['20200107', '20200110', '20200116', '20200312', '20200428', '20200531']

        .. figure:: ../_images/gpr_schema.*
            :name: gpr_schema
            :align: center
            :width: 100%

            Snapshot of the *GPR* schema.
        """

        self.data_dates = [re.findall(r'\d+', x)[0] for x in os.listdir(self.ZIPFILE_DATA_DIR)]

        self.dtypes = {}

        self.db_instance = db_instance

        self.schema_name_ = lambda gpr_date: self.ACRONYM + (gpr_date if gpr_date else '')

    def unzip_backup_data(self, gpr_date, confirmation_required=True, verbose=False):
        """
        Unzip GPR data files (up to the given date) in a local directory.

        :param gpr_date: Date of the data (i.e., name of the data folder).
        :type gpr_date: str
        :param confirmation_required: Whether confirmation is required to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int

        **Examples**::

            >>> from src.preprocessor import GPR
            >>> gpr = GPR()
            >>> gpr.unzip_backup_data(gpr_date='20200531', verbose=True)
            To extract the GPR data of "20200531" from its zipped files
            ? [No]|Yes: yes
            Extraction in progress ...
                Phase 1: extracting the original zipped data files ... Done.
                Phase 2: extracting further the decompressed files ... Done.
        """

        cfm_msg = f"To extract the {self.ACRONYM} data of \"{gpr_date}\" from its zipped files\n?"
        if confirmed(cfm_msg, confirmation_required=confirmation_required):
            if verbose:
                print("Extraction in progress ... ")
                print("\tPhase 1: extracting the original zipped data files", end=" ... ")

            phase_1_archives = glob.glob(cd(self.ZIPFILE_DATA_DIR, gpr_date, "*.zip"))
            if len(phase_1_archives) == 0:
                phase_1_archives = glob.glob(cd(self.ZIPFILE_DATA_DIR, gpr_date + " PM", "*.zip"))

            if len(phase_1_archives) == 0:
                if verbose:
                    print("Failed.")
                return None

            else:
                for archive in phase_1_archives:
                    unzip(archive, cd(self.FILE_DATA_DIR, gpr_date))

                if verbose:
                    print("Done.")

            if verbose:
                print("\tPhase 2: extracting further the decompressed files", end=" ... ")

            phase_2_archives = glob.glob(cd(self.FILE_DATA_DIR, gpr_date, gpr_date[2:], "*.zip"))
            if len(phase_2_archives) == 0:
                phase_2_archives = glob.glob(
                    cd(self.FILE_DATA_DIR, gpr_date, gpr_date[2:] + " PM", "*.zip"))

            if len(phase_2_archives) == 0:
                if verbose:
                    print("Failed.")
                return None

            else:
                for archive in phase_2_archives:
                    unzip(archive, os.path.dirname(os.path.dirname(archive)))
                    # os.remove(archive)
                shutil.rmtree(os.path.commonpath(phase_2_archives))

                if verbose:
                    print("Done.")

    def unzip_data(self, confirmation_required=True, verbose=False):
        """
        Unzip all available GPR data files from the backup in a local directory.

        :param confirmation_required: Whether confirmation is required to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int

        **Examples**::

            >>> from src.preprocessor import GPR
            >>> gpr = GPR()
            >>> gpr.unzip_data(verbose=True)
            To extract all available GPR data from their zipped files
            ? [No]|Yes: yes
            Extraction in progress ...
                "20200107" ... Done.
                "20200110" ... Done.
                "20200116" ... Done.
                "20200312" ... Done.
                "20200428" ... Done.
                "20200531" ... Done.
        """

        cfm_msg = f"To extract all available {self.ACRONYM} data from their zipped files\n?"
        if confirmed(cfm_msg, confirmation_required=confirmation_required):

            print("Extraction in progress ... ")
            for gpr_date in self.data_dates:
                if verbose:
                    print(f"\t\"{gpr_date}\"", end=" ... ")

                try:
                    self.unzip_backup_data(gpr_date, confirmation_required=False, verbose=False)

                    if verbose:
                        print("Done.")

                except Exception as e:
                    _print_failure_msg(e, msg="Failed.")

    @staticmethod
    def parse_gpr_log(path_to_gpr_log, head_tag='ZRDASV2_GPR_Log'):
        """
        Parse a GPR log file (.lxml).

        :param path_to_gpr_log: Path to a log file for GPR data.
        :type path_to_gpr_log: str
        :param head_tag: Name of the top tag; defaults to ``'ZRDASV2_GPR_Log'``.
        :type head_tag: str | None
        :return: Parsed data of the GPR log file.
        :rtype: tuple

        **Examples**::

            >>> from src.preprocessor import GPR
            >>> from pyhelpers.dirs import cd, cdd
            >>> import os
            >>> gpr = GPR()
            >>> # gpr.unzip_data(verbose=True)
            >>> gpr_log_file_paths = []
            >>> for root, dirs, files in os.walk(cd(gpr.FILE_DATA_DIR, "20200531")):
            ...     for filename in files:
            ...         if filename.endswith(".lxml"):
            ...             gpr_log_file_paths.append(cd(root, filename))
            >>> gpr_log_file, chunk_files_info = gpr.parse_gpr_log(gpr_log_file_paths[0])
            >>> type(gpr_log_file)
            collections.OrderedDict
            >>> list(gpr_log_file.keys())
            ['MetaData', 'System_Parameters', 'Operator_Messages', 'ChunkedFiles']
            >>> type(chunk_files_info)
            collections.OrderedDict
            >>> len(chunk_files_info)
            2
        """

        with open(path_to_gpr_log, 'r') as f:
            f.readline()
            gpr_log = xmltodict.parse(f.read())

        if head_tag:
            gpr_log = gpr_log[head_tag]

        chunked_filenames = list(get_dict_values('ReconstitutedName', gpr_log))
        chunked_files_info = list(get_dict_values('Chunk', gpr_log))

        files_catalogue = collections.OrderedDict()
        for cfn, cfi in zip(chunked_filenames, chunked_files_info):
            # noinspection PyUnresolvedReferences
            files_catalogue.update({cfn[0]: cfi})

        return gpr_log, files_catalogue

    def get_files_info(self, gpr_date):
        """
        Get useful information about GPR data files.

        :param gpr_date: Date of the data (i.e. name of the data folder).
        :type gpr_date: str
        :return: Useful information about GPR data files.
        :rtype: dict

        **Examples**::

            >>> from src.preprocessor import GPR
            >>> gpr = GPR()
            >>> # gpr.unzip_backup_data(gpr_date='20200531', verbose=True)
            >>> files_information = gpr.get_files_info(gpr_date='20200531')
            >>> type(files_information)
            dict
            >>> len(files_information)
            5
        """

        files_dir = cd(self.FILE_DATA_DIR, gpr_date)

        gpr_log_file_paths = [
            cd(rt, fn) for rt, _, f in os.walk(files_dir) for fn in f if fn.endswith(".lxml")]

        files_info = {}

        for gpr_log_file_path in gpr_log_file_paths:
            gpr_log, files_catalogue = self.parse_gpr_log(gpr_log_file_path)

            filename_dict = {}
            for k in files_catalogue.keys():
                v = files_catalogue[k]
                if isinstance(v, dict):
                    v = [v]
                filename_dict.update({
                    k.rsplit('.')[1]: [os.path.basename(x['RelativeFilepath']) for x in v]})

            filename_dict.update({'DZX': filename_dict['DZT'][0].replace(".DZT", ".DZX")})

            channels = gpr_log['System_Parameters']['GPRSystems']['GPRSystem']['Channels']['Channel']
            channels = dict(zip(['Channel_{}'.format(x['Name']) for x in channels], channels))

            key = os.path.basename(gpr_log_file_path).rsplit(".")[0]
            files_info.update({key: {'Filenames': filename_dict, 'Channels': channels}})

        return files_info

    # == DZX =====================================================================================

    @staticmethod
    def parse_dzx(path_to_dzx, head_tag='DZX'):
        """
        Parse a DZX file (.DZX).

        :param path_to_dzx: Path to a DZX file for GPR data.
        :type path_to_dzx: str
        :param head_tag: Head tag; defaults to ``'DZX'``.
        :type head_tag: str | None
        :return: Parsed data of the DZX file.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import GPR
            >>> from pyhelpers.dirs import cd
            >>> gpr = GPR()
            >>> # gpr.unzip_backup_data(gpr_date='20200531', verbose=True)
            >>> gpr_files_info = gpr.get_files_info(gpr_date='20200531')
            >>> k1 = list(gpr_files_info.keys())[0]
            >>> dzx_filename = gpr_files_info[k1]['Filenames']['DZX']
            >>> dzx_file_path = cd(gpr.FILE_DATA_DIR, "20200531", dzx_filename)
            >>> dzx_file = gpr.parse_dzx(path_to_dzx=dzx_file_path)
            >>> dzx_file.shape
            (12, 2)
        """

        with open(path_to_dzx, 'r') as f:
            f.readline()
            dzx_dat = xmltodict.parse(f.read())
            f.close()

        if head_tag:
            dzx_dat = dzx_dat[head_tag]

        if 'Macro' in dzx_dat.keys():
            dzx_dat.pop('Macro')

        keys = list(dzx_dat.keys())
        base = dzx_dat[keys[0]]
        if isinstance(base, dict):
            dzx_dat_ = dzx_dat[keys[0]].copy()
        else:
            dzx_dat_ = {keys[0]: dzx_dat[keys[0]]}.copy()
        for k in keys[1:]:
            dzx_dat_.update(dzx_dat[k])

        dzx_data = pd.DataFrame.from_dict(dzx_dat_, orient='index').reset_index()
        dzx_data.columns = ['Name', 'Value']

        return dzx_data

    def read_dzx_by_date(self, gpr_date, update=False, verbose=False):
        """
        Read DZX data for a given date (i.e. data folder).

        :param gpr_date: Date of the data (i.e. name of the data folder).
        :type gpr_date: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :return: DZX data for the given date (i.e. data folder).
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import GPR
            >>> gpr = GPR()
            >>> # gpr.unzip_backup_data(gpr_date='20200531', verbose=True)
            >>> dzx_file_data = gpr.read_dzx_by_date(gpr_date='20200531')
            >>> dzx_file_data.shape
            (60, 6)
        """

        path_to_pickle = cd(self.FILE_DATA_DIR, gpr_date + "_dzx.pkl")

        if os.path.isfile(path_to_pickle) and not update:
            dzx_data = load_data(path_to_pickle)

        else:
            gpr_files_info = self.get_files_info(gpr_date)
            # The length of `gpr_files_info` indicates the number of runs included on the date `gpr_date`

            dzx_data_ = []
            for k, v in gpr_files_info.items():

                path_to_run_pickle = cd(self.FILE_DATA_DIR, gpr_date, k + "_DZX.pkl")

                if os.path.isfile(path_to_run_pickle) and not update:
                    dzx_dat = load_data(path_to_run_pickle)

                else:
                    dzx_fn = v['Filenames']['DZX']
                    dzx_dat = self.parse_dzx(cd(self.FILE_DATA_DIR, gpr_date, dzx_fn))

                    info = [x for x in re.split(r'[_\-@]', k) if x not in ['Project', 'Run']]

                    dzx_dat.insert(0, 'TestTrain', info[0])
                    dzx_dat.insert(1, 'Run', info[3])
                    # datetime.datetime.strptime(info[1] + info[2], '%Y%m%d%H%M%S')
                    dzx_dat.insert(2, 'StartDateTime', pd.to_datetime(info[1] + info[2]))
                    # datetime.datetime.strptime(info[4] + info[5], '%Y%m%d%H%M%S')
                    dzx_dat.insert(3, 'EndDateTime', pd.to_datetime(info[4] + info[5]))

                    save_data(dzx_dat, path_to_run_pickle, verbose=verbose)

                dzx_data_.append(dzx_dat)

            dzx_data = pd.concat(dzx_data_, axis=0, ignore_index=True)

            save_data(dzx_data, path_to_pickle, verbose=verbose)

        return dzx_data

    def read_dzx(self, update=False, verbose=False):
        """
        Read DZX data.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :return: DZX data.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import GPR
            >>> gpr = GPR()
            >>> # gpr.unzip_data(verbose=True)
            >>> gpr_dzx_data = gpr.read_dzx()
            >>> gpr_dzx_data.shape
            (180, 6)
        """

        pickle_filename = f"{min(self.data_dates)}_{max(self.data_dates)}_dzx.pkl"
        path_to_dzx_pickle = cd(self.FILE_DATA_DIR, pickle_filename)

        if os.path.isfile(path_to_dzx_pickle) and not update:
            dzx_data = load_data(path_to_dzx_pickle)

        else:
            if verbose:
                print("Processing ... ")

            dzx_dat_list = []
            for gpr_date in self.data_dates:
                if verbose:
                    print(f"\t'{gpr_date}'", end=" ... ")

                try:
                    dzx_dat_list.append(
                        self.read_dzx_by_date(gpr_date=gpr_date, update=update, verbose=False))

                    if verbose:
                        print("Done.")

                except Exception as e:
                    print(e)

            dzx_data = pd.concat(dzx_dat_list, axis=0, ignore_index=True)

            save_data(dzx_data, path_to_dzx_pickle, verbose=verbose)

        return dzx_data

    def import_dzx(self, update=False, confirmation_required=True, verbose=True, **kwargs):
        """
        Import the DZX data into the project database.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param confirmation_required: Whether confirmation is required to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.import_data`_.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.preprocessor import GPR
            >>> gpr = GPR()
            >>> gpr.import_dzx(if_exists='replace')
            To import DZX data into the table "GPR"."DZX"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/gpr_dzx_tbl.*
            :name: gpr_dzx_tbl
            :align: center
            :width: 100%

            Snapshot of the "GPR"."DZX" table.
        """

        tbl_name = f'"{self.SCHEMA_NAME}"."{self.DZX_TABLE_NAME}"'

        if confirmed(f"To import DZX data into the table {tbl_name}?\n", confirmation_required):

            dzx_data = self.read_dzx(update=update, verbose=verbose)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print(f"Importing the data into {tbl_name}", end=" ... ")

            try:
                self.db_instance.import_data(
                    data=dzx_data, schema_name=self.SCHEMA_NAME, table_name=self.DZX_TABLE_NAME,
                    method=self.db_instance.psql_insert_copy, confirmation_required=False,
                    **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")

    # == DZG =======================================================================================

    @staticmethod
    def parse_dzg(path_to_dzg):
        """
        TODO: Parse a DZG data file.

        :param path_to_dzg: Path to a DZG file.
        :type path_to_dzg: str
        :return: Parsed DZG data (in tabular form).
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import GPR
            >>> from pyhelpers.dirs import cd
            >>> gpr = GPR()
            >>> # gpr.unzip_backup_data(gpr_date='20200531', verbose=True)
            >>> gpr_files_info = gpr.get_files_info(gpr_date='20200531')
            >>> k1 = list(gpr_files_info.keys())[0]
            >>> dzg_filename = gpr_files_info[k1]['Filenames']['DZG'][0]
            >>> dzg_file_path = cd(gpr.FILE_DATA_DIR, "20200531", dzg_filename)
            >>> dzg_dat = gpr.parse_dzg(dzg_file_path)
            >>> type(dzg_dat)
            pandas.core.frame.DataFrame
            >>> dzg_dat.shape
            (2833, 16)
        """

        with open(path_to_dzg, mode='r') as f:
            dzg_file = f.read().strip()
            f.close()

        dzg_data = [x.replace('$GSSIS,', '').replace('\n$GPMAP', '') for x in dzg_file.split('\n\n')]

        csv_dzg_data = csv.reader(dzg_data)

        col_names = ['FileScanNumber', 'ScanCount',
                     'TimeOfPosition',
                     'unknown0',
                     'ELR', 'TrackID', 'Longitude', 'Latitude',
                     'unknown1',
                     'unknown2',
                     'unknown3',
                     'unknown4',
                     'unknown5',
                     'unknown6',
                     'unknown7',
                     'unknown8']

        parsed_dzg_data = pd.DataFrame(csv_dzg_data, columns=col_names)

        return parsed_dzg_data

    def read_dzg_by_date(self, gpr_date, update=False, verbose=False):
        """
        Read DZG data for a given date (i.e. data folder).

        :param gpr_date: Date of the data (i.e. name of the data folder).
        :type gpr_date: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :return: DZG data for the given date (i.e. data folder).
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import GPR
            >>> gpr = GPR()
            >>> # gpr.unzip_backup_data(gpr_date='20200531', verbose=True)
            >>> dzg_file_data = gpr.read_dzg_by_date(gpr_date='20200531')
            >>> dzg_file_data.shape
            (73849, 20)
        """

        path_to_pickle = cd(self.FILE_DATA_DIR, gpr_date + "_dzg.pkl")

        if os.path.isfile(path_to_pickle) and not update:
            dzg_data = load_data(path_to_pickle)

        else:
            gpr_files_info = self.get_files_info(gpr_date)
            # The length of `gpr_files_info` indicates the number of runs
            # included on the date `gpr_date`

            dzg_data_ = []
            for k, v in gpr_files_info.items():

                path_to_run_pickle = cd(self.FILE_DATA_DIR, gpr_date, k + "_DZG.pkl")

                if os.path.isfile(path_to_run_pickle) and not update:
                    dzg_dat = load_data(path_to_run_pickle)

                else:
                    dzg_fn_list = v['Filenames']['DZG']
                    dzg_dat_ = [
                        self.parse_dzg(cd(self.FILE_DATA_DIR, gpr_date, fn)) for fn in dzg_fn_list]

                    if len(dzg_dat_) == 1:
                        dzg_dat = dzg_dat_[0]
                    else:
                        dzg_dat = pd.concat(dzg_dat_, axis=0, ignore_index=True)

                    info = [x for x in re.split(r'[_\-@]', k) if x not in ['Project', 'Run']]

                    dzg_dat.insert(0, 'TestTrain', info[0])
                    dzg_dat.insert(1, 'Run', info[3])
                    # datetime.datetime.strptime(info[1] + info[2], '%Y%m%d%H%M%S')
                    dzg_dat.insert(2, 'StartDateTime', pd.to_datetime(info[1] + info[2]))
                    # datetime.datetime.strptime(info[4] + info[5], '%Y%m%d%H%M%S')
                    dzg_dat.insert(3, 'EndDateTime', pd.to_datetime(info[4] + info[5]))

                    save_data(dzg_dat, path_to_run_pickle, verbose=verbose)

                dzg_data_.append(dzg_dat)

            dzg_data = pd.concat(dzg_data_, axis=0, ignore_index=True)

            save_data(dzg_data, path_to_pickle, verbose=verbose)

        return dzg_data

    def read_dzg(self, update=False, verbose=False):
        """
        Read DZG data.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :return: DZG data.
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import GPR
            >>> gpr = GPR()
            >>> # gpr.unzip_data(verbose=True)
            >>> gpr_dzg_data = gpr.read_dzg()
            >>> gpr_dzg_data.shape
            (517870, 20)
        """

        pickle_filename = f"{min(self.data_dates)}_{max(self.data_dates)}_dzg.pkl"
        path_to_dzg_pickle = cd(self.FILE_DATA_DIR, pickle_filename)

        if os.path.isfile(path_to_dzg_pickle) and not update:
            dzg_data = load_data(path_to_dzg_pickle)

        else:
            if verbose:
                print("Processing ... ")

            dzg_dat_list = []
            for gpr_date in self.data_dates:
                print(f"\t'{gpr_date}'", end=" ... ") if verbose else ""

                try:
                    dzg_dat_list.append(
                        self.read_dzg_by_date(gpr_date=gpr_date, update=update, verbose=False))

                    if verbose:
                        print("Done.")

                except Exception as e:
                    _print_failure_msg(e)

            dzg_data = pd.concat(dzg_dat_list, axis=0, ignore_index=True)

            save_data(dzg_data, path_to_dzg_pickle, verbose=verbose)

        return dzg_data

    def import_dzg(self, update=False, confirmation_required=True, verbose=True, **kwargs):
        """
        Import the DZG data into the project database.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param confirmation_required: Whether confirmation is required to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.import_data`_.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.preprocessor import GPR
            >>> gpr = GPR()
            >>> gpr.import_dzg()
            To import DZG data into the table "GPR"."DZG"?
             [No]|Yes: yes
            Importing the data ... Done.

        .. figure:: ../_images/gpr_dzg_tbl.*
            :name: gpr_dzg_tbl
            :align: center
            :width: 100%

            Snapshot of the "GPR"."DZG" table.
        """

        tbl_name = f"\"{self.SCHEMA_NAME}\".\"{self.DZG_TABLE_NAME}\""

        if confirmed(f"To import DZG data into the table {tbl_name}?\n", confirmation_required):

            dzg_data = self.read_dzg(update=update, verbose=verbose)

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing the data", end=" ... ")

            try:
                self.db_instance.import_data(
                    data=dzg_data, schema_name=self.SCHEMA_NAME, table_name=self.DZG_TABLE_NAME,
                    method=self.db_instance.psql_insert_copy, confirmation_required=False,
                    **kwargs)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_msg(e, msg="Failed.")

    # == DZT =======================================================================================

    @staticmethod
    def parse_dzt_(path_to_dzt):
        """
        TODO: Parse a DZT data file.

        :param path_to_dzt: Path to a DZT file.
        :type path_to_dzt: str
        :return: Parsed DZT data.
        :rtype: list

        **Examples**::

            >>> from src.preprocessor import GPR
            >>> from pyhelpers.dirs import cd
            >>> gpr = GPR()
            >>> # gpr.unzip_backup_data(gpr_date='20200531', verbose=True)
            >>> gpr_files_info = gpr.get_files_info(gpr_date='20200531')
            >>> k1 = list(gpr_files_info.keys())[0]
            >>> dzt_filename = gpr_files_info[k1]['Filenames']['DZT'][0]
            >>> dzt_file_path = cd(gpr.FILE_DATA_DIR, "20200531", dzt_filename)
            >>> dzt_file = gpr.parse_dzt_(path_to_dzt=dzt_file_path)
            >>> type(dzt_file)
            list
        """

        dzt_reader = siina.Radar()

        dzt_reader.read_file(path_to_dzt)

        # # Print dimensions for the data
        # print(f"channels          = {dzt_reader.nchan}\n"
        #       f"samples           = {dzt_reader.ncols}\n"
        #       f"points in samples = {dzt_reader.nrows}")

        dzt_reader.read_markers()  # Strip markers (important step with .DZT files)

        # # Centre each sample (for each trace do func(trace[500:])
        # dzt_reader.func_dc(start=500)
        #
        # # Apply a low-pass filter with cutoff = 6 * frequency
        # dzt_reader.func_filter(cutoff='6')

        return dzt_reader.data_list

    def read_dzt_by_date(self, gpr_date, update=False, pickle_it=False, verbose=False):
        """
        Read DZT data for a given date (i.e. data folder).

        :param gpr_date: Date of the data (i.e. name of the data folder).
        :type gpr_date: str
        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param pickle_it: Whether to save the DZT data as a pickle file; defaults to ``False``.
        :type pickle_it: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``False``.
        :type verbose: bool | int
        :return: DZT data for the given date (i.e. data folder).
        :rtype: list

        **Examples**::

            >>> from src.preprocessor import GPR
            >>> gpr = GPR()
            >>> # gpr.unzip_backup_data(gpr_date='20200531', verbose=True)
            >>> dzt_file_data = gpr.read_dzt_by_date(gpr_date='20200531')
            >>> dzt_file_dat1 = dzt_file_data[0]
            >>> type(dzt_file_dat1)
            dict
            >>> list(dzt_file_dat1.keys())
            ['DZT', 'TestTrain', 'Run', 'StartDateTime', 'EndDateTime']
            >>> type(dzt_file_dat1['DZT'])
            list
            >>> type(dzt_file_dat1['DZT'][0])
            numpy.ndarray
            >>> dzt_file_dat1['DZT'][0].shape
            (512, 9823)
            >>> type(dzt_file_dat1['DZT'][1])
            numpy.ndarray
            >>> dzt_file_dat1['DZT'][1].shape
            (512, 9823)
        """

        path_to_pickle = cd(self.FILE_DATA_DIR, gpr_date + "_dzt.pkl")
        #
        # if os.path.isfile(path_to_pickle) and not update:
        #     dzt_data = load_data(path_to_pickle)
        #
        # else:
        gpr_files_info = self.get_files_info(gpr_date)
        # The length of `gpr_files_info` indicates the number of runs included on the date `gpr_date`

        dzt_data = []
        for k, v in gpr_files_info.items():

            path_to_run_pickle = cd(self.FILE_DATA_DIR, gpr_date, k + "_DZT.pkl")

            if os.path.isfile(path_to_run_pickle) and not update:
                dzt_dat = load_data(path_to_run_pickle)

            else:
                dzt_fn_list = v['Filenames']['DZT']

                temp = []
                for dzt_fn in dzt_fn_list:
                    path_to_dzt = cd(self.FILE_DATA_DIR, gpr_date, dzt_fn)
                    if os.path.isfile(path_to_dzt):
                        f = open(path_to_dzt, mode='rb')
                        temp.append(f.readlines())
                        f.close()
                temp = list(itertools.chain.from_iterable(temp))

                path_to_temp_dzt = cd(self.FILE_DATA_DIR, gpr_date, 'temp.dzt')

                with open(path_to_temp_dzt, 'wb') as f:
                    f.writelines(temp)

                del temp
                gc.collect()

                try:
                    dzt_dat_ = self.parse_dzt_(path_to_temp_dzt)
                except (KeyError, struct.error):
                    dzt_dat_ = [np.empty((512, 0)), np.empty((512, 0))]

                if os.path.isfile(path_to_temp_dzt):
                    os.remove(path_to_temp_dzt)

                info = [x for x in re.split(r'[_\-@]', k) if x not in ['Project', 'Run']]

                dzt_dat = {
                    'DZT': dzt_dat_,
                    'TestTrain': info[0],
                    'Run': info[3],
                    'StartDateTime': datetime.datetime.strptime(info[1] + info[2], '%Y%m%d%H%M%S'),
                    'EndDateTime': datetime.datetime.strptime(info[4] + info[5], '%Y%m%d%H%M%S')
                }

                save_data(dzt_dat, path_to_run_pickle, verbose=verbose)

            dzt_data.append(dzt_dat)

            del dzt_dat
            gc.collect()

        if pickle_it:
            save_data(dzt_data, path_to_pickle, verbose=verbose)

        return dzt_data

    def visualise_dzt_(self, path_to_dzt, channel=0):
        """
        TODO: Visualise DZT data.

        :param path_to_dzt: Path to a DZT file.
        :type path_to_dzt: str
        :param channel: Channel number to visualise; defaults to ``0``.
        :type channel: int
        """

        dzt_data = self.parse_dzt_(path_to_dzt)

        from pyhelpers.settings import mpl_preferences
        mpl_preferences(backend='TkAgg', font_name='Times New Roman')

        import matplotlib.pyplot as plt

        # Plot mean function for the first channel; all channels are found under obj.data_list
        plt.figure(1)
        plt.plot(dzt_data[channel].mean(axis=1))  # to take the mean of each row

        # Plot a radargram with plt.imshow; be careful with the profile size (plt.ncols < 5000)
        plt.figure(2)
        plt.imshow(dzt_data[channel], aspect='auto', cmap='gray')

        # plt.figure(3)
        # plt.plot(dzt_data[1].mean(axis=1))
        # plt.figure(4)
        # plt.imshow(dzt_data[1], aspect='auto', cmap='gray')

        plt.tight_layout()

    def import_dzt_(self, update=False, confirmation_required=True, verbose=True, **kwargs):
        """
        TODO: Import the DZT data into the project database.

        :param update: Whether to reprocess the original data file(s); defaults to ``False``.
        :type update: bool
        :param confirmation_required: Whether a confirmation is required to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information in the console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the method
            `pyhelpers.dbms.PostgreSQL.import_data`_.

        .. _`pyhelpers.dbms.PostgreSQL.import_data`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html
        """

        tbl_name = f"\"{self.SCHEMA_NAME}\".\"{self.DZG_TABLE_NAME}\""

        if confirmed(f"To import {self.DZT_TABLE_NAME} data into the table {tbl_name}?\n",
                     confirmation_required=confirmation_required):

            if self.db_instance is None:
                self.db_instance = TrackFixityDB()

            if verbose:
                print("Importing the data ... ")

            for gpr_date in self.data_dates:
                print(f"\t'{gpr_date}'", end=" ... ")

                dzt_data = self.read_dzt_by_date(update=update, verbose=verbose)

                dzt_data_list = dzt_data[0]['DZT']

                try:
                    self.db_instance.import_data(
                        data=dzt_data_list[0], schema_name=self.SCHEMA_NAME,
                        table_name=self.DZT_TABLE_NAME, method=self.db_instance.psql_insert_copy,
                        confirmation_required=False, **kwargs)

                    if verbose:
                        print("Done.")

                except Exception as e:
                    _print_failure_msg(e, msg="Failed.")
