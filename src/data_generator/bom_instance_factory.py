"""
This file provides functions to generate scheduling problem instances.

Using this file requires a data_generation config. For example, it is necessary to specify
the type of the scheduling problem.
"""
# OS imports
import random
from multiprocessing import Process, Manager
import warnings
import argparse
import os
import json
import itertools
from datetime import datetime

# Config and data handling imports
from src.utils.file_handler.config_handler import ConfigHandler
from src.utils.file_handler.data_handler import DataHandler

# Functional imports
import copy
import tqdm
import numpy as np
from typing import List
from src.data_generator.task import Task
from src.data_generator.sp_factory import SPFactory
from src.agents.heuristic.heuristic_agent import HeuristicSelectionAgent
from src.environments.env_tetris_scheduling import Env

# constants
DEADLINE_HEURISTIC = 'rand'
SEED = 0

def dfs_bom(node, sorted_top, tasks_mapping_ids, deadline, job_index, filename):
    for child in node.get('children', []):
        dfs_bom(child, sorted_top, tasks_mapping_ids, deadline - 1, job_index, filename)
    machines = [0] * 31
    execution_times = {}
    setup_times = {}
    max_runtime = 0
    max_setup = 0
    for machine in node.get('machines', []):
        machines[machine['id']] = 1
        execution_times[machine['id']] = machine['execution_time']
        max_runtime = machine['execution_time'] if max_runtime < machine['execution_time'] else max_runtime
        setup_times[machine['id']] = machine['setup_time']
        max_setup =  machine['setup_time'] if max_setup < machine['setup_time'] else max_setup

    task = Task(job_index=job_index,
            task_index=len(sorted_top),
            task_id=node['operationid'],
            filename=filename,
            parent_index=node['parentid'],
            children=[tasks_mapping_ids[child['operationid']] for child in node.get('children', [])],
            quantity=node['quantity'],
            machines=machines,
            execution_times=execution_times,
            setup_times=setup_times,
            deadline=deadline,
            runtime=max_runtime,
            setup_time=max_setup,
            tools=[],
            _n_tools=0,
            done=0,
            _n_machines=len(machines),
            should_multiply_quantity_to_execution_times=True,
        )
    sorted_top.append(task)
    sorted_top[-1].task_index = len(sorted_top) - 1
    tasks_mapping_ids[node['operationid']] = len(sorted_top) - 1

def get_job_deadline(start_date_str, delivery_date_str):
    # Convert strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S.%f")
    delivery_date = datetime.strptime(delivery_date_str, "%Y-%m-%d %H:%M:%S.%f")

    # Calculate the difference between the two dates as deadline
    deadline = (delivery_date - start_date).total_seconds()
    return deadline

def load_bom_files():
    instance_list: List[List[Task]] = []


    # Define the directory path
#     directory = os.getcwd() + '/data/own_data/ASP-SIMPLE-COMBINE-FRIGORIFICE'
    directory = os.getcwd() + '/data/own_data/ASP-SIMPLE-GEAMURI-TERMOPAN'
#     directory = os.getcwd() + '/data/own_data/ASP-SIMPLE'
#     directory = os.getcwd() + '/data/own_data/ASP-WIDE'
#     directory = os.getcwd() + '/data/own_data/ASP-DEEP'


    # List all files in the directory
    files = os.listdir(directory)

    instance: List[Task] = []
    # Iterate through the files
    for file in files:
        # Check if the file is a regular file (not a directory)
        if file.endswith('.json') and os.path.isfile(os.path.join(directory, file)) :
            # Process the file
            with open(os.path.join(directory, file), 'r') as f:
                bom_job = json.load(f)
                deadline = get_job_deadline(bom_job['start_date'], bom_job['delivery_date'])
                tasks_mapping_ids = dict()
                sorted_top: List[Task] = []
                dfs_bom(bom_job, sorted_top, tasks_mapping_ids, deadline, 0, filename=file)
                for task in sorted_top:
                    if task.parent_index:
                        task.parent_index = tasks_mapping_ids[task.parent_index]
                instance_list.append(sorted_top)
    return instance_list

def main(config_file_name=None, external_config=None):
    # get config
    current_config: dict = ConfigHandler.get_config(config_file_name, external_config)

    # Create instance list
    instance_list: List[List[Task]] = load_bom_files()

#     for job in instance_list:
#         for task in job:
#             print(task)

    # compute individual hash for each instance
    SPFactory.compute_and_set_hashes(instance_list)

    # Write resulting instance data to file
    if current_config.get('write_to_file', False):
        DataHandler.save_instances_data_file(current_config, instance_list)


def get_parser_args():
    """Get arguments from command line."""
    # Arguments for function
    parser = argparse.ArgumentParser(description='Instance generation for scheduling optimization')
    parser.add_argument('-fp', '--config_file_path', type=str, required=True,
                        help='Path to config file you want to use for training')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    # get config_file from terminal input
    parse_args = get_parser_args()
    config_file_path = parse_args.config_file_path

    main(config_file_name=config_file_path)
