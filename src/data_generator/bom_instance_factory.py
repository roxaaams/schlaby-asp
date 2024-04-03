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



def compute_initial_instance_solution(instances: List[List[Task]], config: dict) -> List[List[Task]]:
    """
    Initializes multiple processes (optional) to generate deadlines for the raw scheduling problem instances

    :param instances: List of raw scheduling problem instances
    :param config: Data_generation config

    :return: List of scheduling problems instances with set deadlines

    """
    # Get configured number of processes
    num_processes: int = config.get('num_processes', 1)

    if num_processes > len(instances):
        num_processes = len(instances)
        warnings.warn('num_processes was set to num_instances.'
                      'The number of processes may not exceed the number of instances which need to be generated.',
                      category=RuntimeWarning)

    # Multiprocess case
    manager = Manager()
    instance_list = manager.list()
    make_span_list = manager.list()
    processes = []

    # split instances for multiprocessing
    features_dataset = np.array_split(instances, num_processes)

    for process_id in tqdm.tqdm(range(num_processes), desc="Compute deadlines"):
        args = (features_dataset[process_id], instance_list, make_span_list, config)
        p = Process(target=generate_deadlines, args=args)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    return list(instance_list)


def generate_deadlines(instances: List[List[Task]], instance_with_dead_lines: List[List[Task]],
                       make_span_list: List[List[int]], config: dict) -> None:
    """
    Generates suitable deadlines for the input instances

    :param instances: List of raw scheduling problem instances
    :param instance_with_dead_lines: manager.list() (Only in Multi-process case)
    :param make_span_list: manager.list() (Only in Multi-process case)
    :param config: Data_generation config

    :return: None

    """
    heuristic_agent = HeuristicSelectionAgent()
    make_span = []
    np.random.seed(config.get('seed', SEED))
    for i, instance in enumerate(instances):
        # create env
        env = Env(config, [instance])

        done = False
        total_reward = 0
        t = 0
        runtimes = [task.runtime for task in instance]
        # run agent on environment and collect rewards until done
        while not done:
            tasks = env.tasks
            task_mask = env.get_action_mask()

            action = heuristic_agent(tasks, task_mask, DEADLINE_HEURISTIC)
            b = env.step(action)
            total_reward += b[1]
            done = b[2]
            t += 1

        tasks = env.tasks

        # start_times = env.scheduling
        make_span.append(env.get_makespan())
        # actions.sort()
        for task_j, task in enumerate(tasks):
            task.deadline = task.finished
            task._deadline = task.finished
            task.runtime = runtimes[task_j]
            task._run_time_left = runtimes[task_j]
            task.running = 0
            task.done = 0
            task._started_in_generation = copy.copy(task.started)
            task.started = 0
            task.finished = 0
            task._optimal_machine = int(task.selected_machine)

        instance_with_dead_lines.append(tasks)
        make_span_list.append(make_span)


def dfs_bom(node, sorted_top, tasks_mapping_ids, deadline, job_index):
    for child in node.get('children', []):
        dfs_bom(child, sorted_top, tasks_mapping_ids, deadline - 1, job_index)
#     if node['parentid']:
#         parent_index =  node['parentid']
#     else:
#         parent_index = None
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
#             parent_index=parent_index,
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
            _n_machines=len(machines),
            done=False,
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
    directory = os.getcwd() + '/data/own_data/ASP-WIDE'

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
                dfs_bom(bom_job, sorted_top, tasks_mapping_ids, deadline, 0)
#                 for task in job:
#                     task.parent_index = tasks_mapping_ids[task.parent_index]
                instance_list.append(sorted_top)
    return instance_list

def main(config_file_name=None, external_config=None):
    # get config
    current_config: dict = ConfigHandler.get_config(config_file_name, external_config)

    # Create instance list
    instance_list: List[List[Task]] = load_bom_files()

#     # Assign deadlines in-place
#     SPFactory.set_deadlines_to_max_deadline_per_job(instance_list, current_config.get('num_jobs', None))
# #
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
