"""
This file provides functions to handle the import and export of bomfiles (e.g. training bomfile).

Constants:
bomfile_DIRECTORY: Path: Parent directory for the bomfile import
FILE_PATH_VARIABLES: list[str]: Necessary variables to load bomfile by dict
DIRECTORY_PATH_VARIABLES: dict[str, str]: Dict to specify whether training, testing or data_generation should be loaded

SCHEMA_FILE: str: Schema file for checking the loaded bomfiles
"""
# OS imports
import os
from pathlib import Path
from typing import List, Dict
import warnings
import operator

# Data handling imports
import yaml
import json
import jsonschema

# Constants
bomfile_DIRECTORY: Path = Path(__file__).parent.parent.parent.parent / 'config'
DIRECTORY_PATH_VARIABLES: Dict[str, str] = {'data_generation': 'sp_type', 'training': 'algorithm',
                                            'testing': 'test_algorithm'}



class FileHandler:
    """
    Handles the import of bomfiles.
    """

    @classmethod
    def get_bomfile_from_path(cls, bomfile_file_path: str, check_against_schema: bool = True) -> Dict:
        """
        Initializes the loading of a bomfile

        :param bomfile_file_path: Relative path to a bomfile file (e.g. training/dqn/bomfile_job3_task4_tools0.yaml) which
            was entered to the terminal
        :param check_against_schema: Checking against schema is activated per default (True), can be deactivated
            by setting to False

        :return: bomfile dict

        """

        # Load bomfile
        with open(bomfile_path, 'r') as stream:
            current_bomfile = yaml.load(stream, Loader=yaml.Loader)



        return current_bomfile

    @classmethod
    def get_bomfile(cls, bomfile_file_path=None, external_bomfile=None):
        """
        Gets a bomfile from file or uses external bomfile, according to input

        :param bomfile_file_path: Path to the bomfile file
        :param external_bomfile: bomfile which was created or loaded in an external script
        :param check_against_schema: Checking against schema is activated per default (True), can be deactivated
            by setting to False

        :return: bomfile dictionary

        """
#         assert bool(bomfile_file_path) != bool(external_bomfile), \
#             'You either have to specify a path to the bomfile you want to use for' \
#             'OR provide a pass a loaded bomfile to this function'
        if bomfile_file_path is not None:
            bomfile_file_path_long = Path(bomfile_file_path)
            # create full path
            result = bomfile_DIRECTORY / bomfile_file_path_long
            # Only possible if bomfile file exists
            assert result.exists(), f"Path {result} not found. " \
                                         f"You need to point to a bomfile in accordance to your settings in the bomfile " \
                                         f"folder"

        else:
            bomfile = external_bomfile

        return result





