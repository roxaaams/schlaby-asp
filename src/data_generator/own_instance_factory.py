
# OS imports
import random
from multiprocessing import Process, Manager
import warnings
import argparse

# Config and data handling imports
from src.utils.file_handler.config_handler import ConfigHandler
from src.utils.file_handler.file_handler import FileHandler

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

import torch
import numpy as np
import json

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


def get_topological_order(index, node, visited, tasks, deadline):
    visited.add(node['operationid'])
    if not node['parentid']:
        for maintenance in node.get('maintenances', []):
                task = Task(
                    job_index=index,
                    task_index=maintenance['id'],#len(result),#child['operationid'], #len(result),
                    processing_times=[2],
                    runtime=2,
                    machines=[maintenance['machine_id']],
                    _n_tools=0,
                    done=False,
                    type='maintenance',
                )
                tasks.append(task)

    for child in node.get('children', []):
#         print(child['machines'])
        if child['operationid'] not in visited:
            get_topological_order(index, child, visited, tasks, deadline-1)

    machines = [machine['id'] for machine in node['machines']]
    task = Task(
        job_index=index,
        task_index=node['operationid'],#len(result),#child['operationid'], #len(result),
        parentId=node['parentid'],
        quantity=node['quantity'],
        processing_times=[machine['execution_time'] for machine in node['machines']],
        setup_times=[machine['setup_time'] for machine in node['machines']],
        children=[child['operationid'] for child in node.get('children', [])],
        runtime=2,
        machines=machines,
        tools=[],
        _n_machines=len(machines),
        _n_tools=0,
        deadline=deadline,
        done=False,
        type='op',
    )
    tasks.append(task)




# id_op [(machine_id, proc_time, setup_time)...] [lista de predecesori] lista de succesor [e 1 oricum]

def load_asp(index, deadline, filename, num_mas, num_opes):
    '''
        Load the local ASP instance.

    '''

    data = None

    with open(filename, 'r') as f:
        data = json.load(f)
    # Now 'data' is a Python dictionary containing the parsed JSON data
    # print(data)

    # Call the function for each node in the tree
    visited_nodes = set()
    topological_order = []
    operations: List[Task] = []
    get_topological_order(index, data, visited_nodes, operations, deadline)

#     for node in data['children']:
#         if node['operationid'] not in visited_nodes:
#             get_topological_order(index, node, visited_nodes, topological_order, operations, deadline)
#         # The 'topological_order' list now contains the topological order of the tree nodes

    return topological_order, operations


def main(config_file_name=None, boms_folder = None, external_config=None):

    # get config
    current_config: dict = ConfigHandler.get_config(config_file_name, external_config)

    # get generated instances initial
    generated_instances: List[List[Task]] = []


    index = 0
    deadline = 100


     # get bom_filename
    bom_filename = FileHandler.get_bomfile('data_generation/asp/bom_2_10_10_5.json', 'data_generation/asp/random_bom.json')
    topological_order, operations = load_asp(index, deadline, bom_filename, 10, 10)
    generated_instances.append(operations)



    # Create instance list
    instance_list: List[List[Task]] = generated_instances #compute_initial_instance_solution(generated_instances, current_config)

#     for instance in instance_list:
#         for task_j, task in enumerate(instance):
#             task._deadline = task.finished
#             task._run_time_left = runtimes[task_j]
#             task.running = 0
#             task.done = 0
#             task._started_in_generation = 0,
#             task.started = 0
#             task.finished = 0
#             task._optimal_machine = int(task.selected_machine)


    # Assign deadlines in-place
    SPFactory.set_deadlines_to_max_deadline_per_job(instance_list, current_config.get('num_jobs', None))

    # print("instance_list before hashes", instance_list)


    for inner_list in instance_list:
        # Iterate through the inner list
        for item in inner_list:
#             print(item.str_info_more())
            print(item.task_index)

    # compute individual hash for each instance
    SPFactory.compute_and_set_hashes(instance_list)


    # Write resulting instance data to file
    if current_config.get('write_to_file', False):
        print("'write_to_file', False")
        DataHandler.save_instances_data_file(current_config, instance_list)


def get_parser_args():
    """Get arguments from command line."""
    # Arguments for function
    parser = argparse.ArgumentParser(description='Instance generation for scheduling optimization')
    parser.add_argument('-fp', '--config_file_path', type=str, required=True,
                        help='Path to config file you want to use for training')

    parser.add_argument('-bf', '--boms_folder', type=str, required=True,
                        help='Path to boms_folder you want to use for training')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    # get config_file from terminal input
    parse_args = get_parser_args()
    config_file_path = parse_args.config_file_path
    boms_folder = parse_args.boms_folder

    main(config_file_name=config_file_path, boms_folder=boms_folder)
