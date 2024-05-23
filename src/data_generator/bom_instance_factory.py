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
import random
import copy


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
SHOULD_GENERATE_SIMILAR_INSTANCES = True
SIMILAR_INSTANCES_NUMBER = 496

def dfs_bom(node, sorted_top, tasks_mapping_ids, deadline, job_index, filename, quantity):
    for child in node.get('children', []):
        dfs_bom(child, sorted_top, tasks_mapping_ids, deadline - 1, job_index, filename, quantity * node['quantity'])
    machines = [0] * 31
    execution_times = {}
    setup_times = {}
    max_runtime = 0
    max_setup = 0
    average_runtime = 0
    for machine in node.get('machines', []):
        machines[machine['id']] = 1
        execution_times[machine['id']] = machine['execution_time']
        max_runtime = machine['execution_time'] if max_runtime < machine['execution_time'] else max_runtime
        setup_times[machine['id']] = machine['setup_time']
        max_setup =  machine['setup_time'] if max_setup < machine['setup_time'] else max_setup
        average_runtime = average_runtime + machine['execution_time']
    average_runtime = int(average_runtime / len(node.get('machines', [])))

    task = Task(job_index=job_index,
            task_index=len(sorted_top),
            task_id=node['operationid'],
            filename=filename,
            parent_index=node['parentid'],
            children=[tasks_mapping_ids[child['operationid']] for child in node.get('children', [])],
            quantity=node['quantity'] * quantity,
            machines=machines,
            execution_times=execution_times,
            setup_times=setup_times,
            deadline=deadline,
            runtime=max_runtime,
            average_runtime=average_runtime,
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
#     directory = os.getcwd() + '/data/own_data/ASP-COMBINE-FRIGORIFICE-MAI-PUTINE-MASINI'
#     directory = os.getcwd() + '/data/own_data/ASP-SIMPLE-GEAMURI-TERMOPAN'
#     directory = os.getcwd() + '/data/own_data/ASP-SIMPLE'
    directory = os.getcwd() + '/data/own_data/ASP-WIDE'
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
                dfs_bom(bom_job, sorted_top, tasks_mapping_ids, deadline, 0, filename=file, quantity=1)
                for task in sorted_top:
                    if task.parent_index:
                        task.parent_index = tasks_mapping_ids[task.parent_index]
                instance_list.append(sorted_top)

    if SHOULD_GENERATE_SIMILAR_INSTANCES is True:
        original_list_length = len(instance_list)
#         for i in range(original_list_length):
#             for task in instance_list[i]:
#                  max_runtime = 0
#                  max_setup = 0
#                  average_runtime = 0
#                  for machine_id in range(len(task.machines)):
#                     machine_op_type = random.randint(0, 1)
#                     # 0- should add new machine if not existing, 1 - should modify current current machine times
#                     if machine_op_type == 0:
#                         if task.machines[machine_id] == 0:
#                             task.machines[machine_id] = 1
#                             task.execution_times[machine_id] = random.randint(10, task.runtime)
#                             task.setup_times[machine_id] = random.randint(10, task.setup_time)
#                     else:
#                         task.execution_times[machine_id] = random.randint(10, task.runtime)
#                         task.setup_times[machine_id] = random.randint(10, task.setup_time)
#                     max_runtime = max(max_runtime, task.execution_times[machine_id])
#                     max_setup = max(max_setup, task.setup_times[machine_id])
#                     average_runtime += task.execution_times[machine_id]
#                  task.runtime = max_runtime
#                  task.average_runtime = int(average_runtime / len(task.machines))
#                  task.setup_time = max_setup
        for i in range(SIMILAR_INSTANCES_NUMBER):
            instance_index = random.randint(0, original_list_length - 1)
            new_instance = copy.deepcopy(instance_list[instance_index])
            for task in new_instance:
                max_runtime = 0
                max_setup = 0
                average_runtime = 0
                for machine_id in range(len(task.machines)):
                   machine_op_type = random.randint(0, 1)
                   # 0- should add new machine if not existing, 1 - should modify current current machine times
                   if machine_op_type == 0:
                       if task.machines[machine_id] == 0:
                           task.machines[machine_id] = 1
                           task.execution_times[machine_id] = random.randint(10, task.runtime)
                           task.setup_times[machine_id] = random.randint(10, task.setup_time)
                   else:
                        task.execution_times[machine_id] = random.randint(10, task.runtime)
                        task.setup_times[machine_id] = random.randint(10, task.setup_time)
                   max_runtime = max(max_runtime, task.execution_times[machine_id])
                   max_setup = max(max_setup, task.setup_times[machine_id])
                   average_runtime += task.execution_times[machine_id]
                task.runtime = max_runtime
                task.average_runtime = int(average_runtime / len(task.machines))
                task.setup_time = max_setup

            instance_list.append(new_instance)

    return instance_list

def main(config_file_name=None, external_config=None):
    # get config
    current_config: dict = ConfigHandler.get_config(config_file_name, external_config)

    # Create instance list
    instance_list: List[List[Task]] = load_bom_files()

#     for job in instance_list:
#         for task in job:
#             print(task.quantity, task.parent_index, task.task_index, task.task_id, task.runtime, task.average_runtime)

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
