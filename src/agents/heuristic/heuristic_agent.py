"""
This module provides the following scheduling heuristics as function:

- EDD: earliest due date
- SPT: shortest processing time first
- MTR: most tasks remaining
- LTR: least tasks remaining
- Random: random action

You can implement additional heuristics in this file by specifying a function that takes a list of tasks and an action
mask and returns the index of the job to be scheduled next.

If you want to call your heuristic via the HeuristicSelectionAgent or edit an existing shortcut,
adapt/extend the task_selection dict attribute of the HeuristicSelectionAgent class.

:Example:

Add a heuristic that returns zeros (this is not a practical example!)
1. Define the according function

.. code-block:: python

    def return_0_heuristic(tasks: List[Task], action_mask: np.array) -> int:
        return 0

2. Add the function to the task_selection dict within the HeuristicSelectionAgent class:

.. code-block:: python

    self.task_selections = {
        'rand': random_task,
        'EDD': edd,
        'SPT': spt,
        'MTR': mtr,
        'LTR': ltr,
        'ZERO': return_0_heuristic
    }

"""
import numpy as np
from typing import List
import random
import copy


from src.data_generator.task import Task

def get_active_task_dict(tasks: List[Task]) -> dict:
    """
    Helper function to determining the next unfinished task to be processed for each job

    :param tasks: List of task objects, so one instance

    :return: Dictionary containing the next tasks to be processed for each job

    Would be an empty dictionary if all tasks were completed

    """
    active_job_task_dict = {}
    for task_i, task in enumerate(tasks):
        if not task.done and task.job_index not in active_job_task_dict.keys():
            active_job_task_dict[task.job_index] = task_i

    return active_job_task_dict

def get_active_task_dict_asp(tasks: List[Task]) -> dict:
    """
    Helper function to determining the next unfinished task to be processed for each job

    :param tasks: List of task objects, so one instance

    :return: Dictionary containing the next tasks to be processed for each job

    Would be an empty dictionary if all tasks were completed

    """
    active_task_dict = {}
    for task_i, task in enumerate(tasks):
        are_children_done = True
        for task_j, sub_task in enumerate(tasks):
            if sub_task.task_index in task.children:
                if not sub_task.done:
                    are_children_done = False

        if not task.done and are_children_done is True and task.task_index not in active_task_dict.keys():
            active_task_dict[task.task_index] = 1

    return active_task_dict

def is_leaf(task: Task):
    return len(task.children) == 0 and task.parent_index

critical_path = ([], 0)

def compute_paths(tasks: List[Task], task: Task, path, duration, visited):
    global critical_path
    path.append(task.task_index)

    # Compute the length (cumulative processing # time) of each path determined in step 4.1.
    duration += task.max_execution_times_setup

    # 4.3 a: Determine the critical (the largest cumulative processing time) path
    if is_leaf(task) and duration > critical_path[1]:
        critical_path = (copy.deepcopy(path), duration)
        return

    for _, index_subtask in enumerate(task.children):
        compute_paths(tasks, tasks[index_subtask], path, duration, visited)
        path.pop()

def letsa(tasks: List[Task], action_mask: np.array, feasible_tasks, visited, max_deadline):
    global critical_path
    critical_path = ([], 0)
    length = len(feasible_tasks)
    print(length)
    max_path_length_task_index = feasible_tasks[0]
    max_path_length_delete_index = 0
    compute_paths(tasks, tasks[max_path_length_task_index], [], 0, visited)
    max_length_critical_path = (critical_path[0].copy(), critical_path[1])

    for i in range(len(feasible_tasks)):
        critical_path = ([], 0)
        compute_paths(tasks, tasks[feasible_tasks[i]], [], 0, visited)
        if max_length_critical_path[1] < critical_path[1]:
            max_path_length_task_index  = feasible_tasks[i]
            max_path_length_delete_index = i
            max_length_critical_path = (critical_path[0].copy(), critical_path[1])

    # print('Length of critical path', max_length_critical_path[1])
    # for i in range(len(max_length_critical_path[0])):
    #     print('Task_index:', max_length_critical_path[0][i], 'Task_id in BOM: ', tasks[max_length_critical_path[0][i]].task_id, ' Quantity: ', tasks[max_length_critical_path[0][i]].quantity, ' Runtime: ', tasks[max_length_critical_path[0][i]].max_execution_times_setup)

    # 4.3 b Select the operation Je of the critical path that also belongs to the feasible list F.
    # in this case it is the first operation, which also belongs to F, that is selected for scheduling.

    # 4.4 Set its tentative completion time Ce equal to: (i) the starting time of operation Je from the
    # partial schedule (constraint 2.2 of (PI)), (ii) the due-date De if operation c is the last
    # operation of the final assembly Pe (constraint 2.4 of (PI)).
    completion_time = 0
    if not tasks[max_path_length_task_index].parent_index:
        completion_time = max_deadline
    else:
        # ??? Choose the earliest starting time of all successors Jc
        completion_time = max_deadline
        parent_start_task_index = tasks[max_path_length_task_index].parent_index
        if parent_start_task_index and tasks[parent_start_task_index].done and tasks[parent_start_task_index].started < completion_time:
            completion_time = tasks[parent_start_task_index].started # - EPSILON
        # else:
        #     print('Else branch', parent_start_task_index, tasks[parent_start_task_index].done, tasks[parent_start_task_index].started, completion_time)
        # if tasks[parent_start_task_index].started > completion_time:
        #     print('parent_start_task_index', parent_start_task_index, 'Completion time:', completion_time, 'Start time:', tasks[parent_start_task_index].started)


    # 4.5 Compute the starting time based on the available machines that can produce the operation Jc
    # 4.6 Schedule operation Jc at the latest available starting time Sc on the corresponding machine

    # #  NOTE: instead of scheduling right away, skip it now, follow the next steps and only at the end return the index of the start of operation to be scheduled
    # # 4.7 Delete operation Jc from the operation network.
    # tasks[start_task_index].deleted = True
    # # 4.8 Add all operations Ji such that di = Jc, to the feasible list.
    # # Also check is the operation was not added in the list
    # # Priority is given to the predecessor in the critical path while updating the list of feasible operations
    del feasible_tasks[max_path_length_delete_index]
    for _, sub_task_index in enumerate(tasks[max_path_length_task_index].children):
        if not tasks[sub_task_index].done:
            feasible_tasks.append(sub_task_index)
    # print('Task: ', tasks[max_path_length_task_index].task_id, ' completion_time: ', completion_time)
    # print('FEASIBLE TASKS: ')
    # for i in range(len(feasible_tasks)):
    #     print(tasks[feasible_tasks[i]].task_id, tasks[tasks[feasible_tasks[i]].parent_index].task_id, tasks[tasks[feasible_tasks[i]].parent_index].started)
    return max_path_length_task_index, int(completion_time)


def edd_asp(tasks: List[Task], action_mask: np.array) -> int:
    """
    EDD: earliest due date. Determines the task with the smallest deadline

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the task selected according to the heuristic

    """

    possible_tasks = get_active_task_dict_asp(tasks)
    task_index = -1
    earliest_due_date = np.inf
    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys() and task.deadline < earliest_due_date and not task.done:
            earliest_due_date = task.deadline
            task_index = i
    return task_index

def edd_asp_first_k(tasks: List[Task], k: int = 3):
    """
    EDD: earliest due date. Determines the first k tasks with the smallest deadlines

    :param tasks: List of task objects, so one instance
    :param k: Number of tasks to select with the smallest deadlines.

    :return: Index of the task selected according to the heuristic

    """

    possible_tasks = get_active_task_dict_asp(tasks)
    task_deadlines = []

    # Collect tasks with their deadlines if they are possible and not done
    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys() and not task.done:
            task_deadlines.append((i, task.deadline))

    # Sort tasks by deadline and select the first k
    task_deadlines.sort(key=lambda x: x[1])
    selected_tasks = [task[0] for task in task_deadlines[:k]]

    return selected_tasks

def ect_asp(tasks: List[Task], ends_of_machine_occupancies) -> int:
    """
        ECT: Earliest Completion Time. Determines the first  task with the earliest completion times.

     :param ends_of_machine_occupancies:
     :param tasks: List of task objects, so one instance.

     :return: Index of the tasks selected according to the heuristic.
     """
    possible_tasks = get_active_task_dict_asp(tasks)
    selected_task_i = -1
    earliest_completion_time = np.inf
    print('ends_of_machine_occupancies:', ends_of_machine_occupancies)
    # for i, task in enumerate(tasks):
    #     if task.task_index in possible_tasks.keys() and not task.done:
    #        for machine_id in range(len(tasks[selected_task_i].machines)):
    #             if tasks[selected_task_i].machines[machine_id] == 1:
                    # completion_time = ends_of_machine_occupancies[machine_id] + task.execution_times_setup[machine_id]
                    # if completion_time < earliest_completion_time:
                    #     earliest_completion_time = completion_time
                    #     selected_task_i = i
    return selected_task_i

def ect_asp_first_k(tasks: List[Task], ends_of_machine_occupancies=None, k: int = 3):
    """
    ECT: Earliest Completion Time. Determines the first k tasks with the earliest completion times.

    :param ends_of_machine_occupancies:
    :param tasks: List of task objects, so one instance.
    :param k: Number of tasks to select with the earliest completion times.

    :return: List of indices of the tasks selected according to the heuristic.
    """
    if ends_of_machine_occupancies is None:
        ends_of_machine_occupancies = {}
    possible_tasks = get_active_task_dict_asp(tasks)
    task_completion_times = []

    # Collect tasks with their completion times if they are possible and not done
    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys() and not task.done:
            for machine_id in range(len(task.machines)):
                if task.machines[machine_id] == 1:
                    completion_time = ends_of_machine_occupancies[machine_id] + task.execution_times_setup[machine_id]
                    task_completion_times.append((i, completion_time))

    # Sort tasks by completion time and select the first k
    task_completion_times.sort(key=lambda x: x[1])
    selected_tasks = [task[0] for task in task_completion_times[:k]]

    return selected_tasks


def mpo_root_asp(tasks: List[Task], action_mask: np.array) -> int:
    possible_tasks = get_active_task_dict_asp(tasks)
    task_index = 0
    max_number_children = -1
    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys():
            remaining_tasks_count = 0
            task_successor_index = task.parent_index
            while task_successor_index is not None:
                remaining_tasks_count += 1
                task_successor_index = tasks[task_successor_index].parent_index
            if max_number_children < remaining_tasks_count:
                max_number_children = remaining_tasks_count
                task_index = task.task_index
    return task_index

def mpo_root_asp_first_k(tasks: List[Task], k: int) -> List[int]:
    """
    MPO (Maximal Predecessor Operations Rule): Determines the first k tasks with the highest number of predecessors, starting from the root.
    :param tasks: List of task objects, so one instance.
    :param k: Number of tasks to select with the highest number of predecessors.
    :return: List of indices of the tasks selected according to the heuristic.
    """
    possible_tasks = get_active_task_dict_asp(tasks)
    task_predecessors = []

    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys():
            remaining_tasks_count = 0
            task_successor_index = task.parent_index
            while task_successor_index is not None:
                remaining_tasks_count += 1
                task_successor_index = tasks[task_successor_index].parent_index
            task_predecessors.append((i, remaining_tasks_count))

    # Sort tasks by the number of predecessors in descending order
    task_predecessors.sort(key=lambda x: x[1], reverse=True)

    # Select the first k tasks
    selected_tasks = [task[0] for task in task_predecessors[:k]]

    return selected_tasks


def mpo_asp(tasks: List[Task], action_mask: np.array) -> int:
    """
    MPO (Maximal Predecessor Operations Rule): Determines the task with the highest number of predecessors.

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the task selected according to the heuristic

    """

    possible_tasks = get_active_task_dict_asp(tasks)
    task_index = 0
    max_number_children = 0
    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys():
            if len(task.children) >= max_number_children:
                max_number_children = len(task.children)
                task_index = task.task_index
    return task_index

def mpo_asp_first_k(tasks: List[Task], k: int) -> List[int]:
    """
    MPO (Maximal Predecessor Operations Rule): Determines the first k tasks with the highest number of predecessors.

    :param tasks: List of task objects, so one instance.
    :param k: Number of tasks to select with the highest number of predecessors.

    :return: List of indices of the tasks selected according to the heuristic.
    """
    possible_tasks = get_active_task_dict_asp(tasks)
    task_predecessors = []

    # Collect tasks with their number of predecessors if they are possible
    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys():
            task_predecessors.append((i, len(task.children)))

    # Sort tasks by the number of predecessors in descending order and select the first k
    task_predecessors.sort(key=lambda x: x[1], reverse=True)
    selected_tasks = [task[0] for task in task_predecessors[:k]]

    return selected_tasks

# def mpo_asp_standard(tasks: List[Task], action_mask: np.array) -> int:
#     """
#     MPO (Maximal Predecessor Operations Rule): Determines the task with the highest number of predecessors (not just the direct ones).
#
#     :param tasks: List of task objects, so one instance
#     :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic
#
#     :return: Index of the task selected according to the heuristic
#
#     """
#
#     possible_tasks = get_active_task_dict_asp(tasks)
#     task_index = 0
#     max_number_children = 0
#     for i, task in enumerate(tasks):
#         if task.task_index in possible_tasks.keys():
#             if task.total_subnodes >= max_number_children:
#                 max_number_children = task.total_subnodes
#                 task_index = task.task_index
#     return task_index


# def lpo_asp_standard(tasks: List[Task], action_mask: np.array) -> int:
#     """
#     LPO (Least Predecessor Operations Rule): Determines the task with the least number of predecessors (not just the direct ones).
#
#     :param tasks: List of task objects, so one instance
#     :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic
#
#     :return: Index of the task selected according to the heuristic
#
#     """
#
#     possible_tasks = get_active_task_dict_asp(tasks)
#     task_index = 0
#     min_number_children = np.inf
#     for i, task in enumerate(tasks):
#         if task.task_index in possible_tasks.keys():
#             if task.total_subnodes < min_number_children:
#                 min_number_children = len(task.children)
#                 task_index = task.task_index
#     return task_index

def lpo_asp(tasks: List[Task], action_mask: np.array) -> int:
    """
    LPO (Least Predecessor Operations Rule): Determines the task with the least number of predecessors.

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the task selected according to the heuristic

    """

    possible_tasks = get_active_task_dict_asp(tasks)
    task_index = 0
    min_number_children = np.inf
    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys():
            if len(task.children) < min_number_children:
                min_number_children = len(task.children)
                task_index = task.task_index
    return task_index

def lpo_asp_first_k(tasks: List[Task], k: int) -> List[int]:
    """
    LPO (Least Predecessor Operations Rule): Determines the first k tasks with the least number of predecessors.

    :param tasks: List of task objects, so one instance.
    :param k: Number of tasks to select with the least number of predecessors.

    :return: List of indices of the tasks selected according to the heuristic.
    """
    possible_tasks = get_active_task_dict_asp(tasks)
    task_predecessors = []

    # Collect tasks with their number of predecessors if they are possible
    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys():
            task_predecessors.append((i, len(task.children)))

    # Sort tasks by the number of predecessors in ascending order and select the first k
    task_predecessors.sort(key=lambda x: x[1])
    selected_tasks = [task[0] for task in task_predecessors[:k]]

    return selected_tasks


def lpo_root_asp(tasks: List[Task], action_mask: np.array) -> int:
    """
    LPO (Least Predecessor Operations Rule): Determines the task with the least number of predecessors, starting from the root.
    :param tasks:
    :param action_mask:
    :return:
    """
    possible_tasks = get_active_task_dict_asp(tasks)
    task_index = 0
    min_number_children = np.inf
    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys():
            remaining_tasks_count = 0
            task_successor_index = task.parent_index
            while task_successor_index is not None:
                remaining_tasks_count += 1
                task_successor_index = tasks[task_successor_index].parent_index
            if min_number_children > remaining_tasks_count:
                min_number_children = remaining_tasks_count
                task_index = task.task_index
    return task_index

def lpo_root_asp_first_k(tasks: List[Task], k: int) -> List[int]:
    """
    LPO (Least Predecessor Operations Rule): Determines the first k tasks with the least number of predecessors, starting from the root.
    :param tasks: List of task objects, so one instance.
    :param k: Number of tasks to select with the least number of predecessors.
    :return: List of indices of the tasks selected according to the heuristic.
    """
    possible_tasks = get_active_task_dict_asp(tasks)
    task_predecessors = []

    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys():
            remaining_tasks_count = 0
            task_successor_index = task.parent_index
            while task_successor_index is not None:
                remaining_tasks_count += 1
                task_successor_index = tasks[task_successor_index].parent_index
            task_predecessors.append((i, remaining_tasks_count))

    # Sort tasks by the number of predecessors in ascending order
    task_predecessors.sort(key=lambda x: x[1])

    # Select the first k tasks
    selected_tasks = [task[0] for task in task_predecessors[:k]]

    return selected_tasks

def spt_asp(tasks: List[Task], action_mask: np.array) -> int:
    """
    SPT: shortest processing time first. Determines the unfinished task has the lowest runtime for ASP

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the task selected according to the heuristic

    """
    possible_tasks = get_active_task_dict_asp(tasks)
    task_index = -1
    shortest_processing_time = np.inf
    for i, task in enumerate(tasks):
        if not task.done and task.task_index in possible_tasks.keys():
            if task.max_execution_times_setup < shortest_processing_time:
                shortest_processing_time = task.max_execution_times_setup
                task_index = i
    return task_index

def spt_asp_first_k(tasks: List[Task], k: int) -> List[int]:
    """
    SPT: shortest processing time first. Determines the first k unfinished tasks with the lowest runtime for ASP.

    :param tasks: List of task objects, so one instance.
    :param k: Number of tasks to select with the shortest processing times.

    :return: List of indices of the tasks selected according to the heuristic.
    """
    possible_tasks = get_active_task_dict_asp(tasks)
    task_runtimes = []

    # Collect tasks with their runtimes if they are possible and not done
    for i, task in enumerate(tasks):
        if not task.done and task.task_index in possible_tasks.keys():
            task_runtimes.append((i, task.max_execution_times_setup))

    # Sort tasks by runtime in ascending order and select the first k
    task_runtimes.sort(key=lambda x: x[1])
    selected_tasks = [task[0] for task in task_runtimes[:k]]

    return selected_tasks

def random_task_asp(tasks: List[Task], action_mask: np.array) -> int:
    """
    Returns a random task

    :param tasks: Not needed
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    possible_tasks = get_active_task_dict_asp(tasks)
    random_task = random.choice(list(possible_tasks.keys()))
    return random_task


def random_task_asp_first_k(tasks: List[Task], k: int) -> List[int]:
    """
    Returns k random tasks.

    :param tasks: Not needed.
    :param k: Number of random tasks to select.

    :return: List of indices of the jobs selected according to the heuristic.
    """
    possible_tasks = list(get_active_task_dict_asp(tasks).keys())
    if len(possible_tasks) < k:
        raise ValueError("Number of possible tasks is less than k.")
    random_tasks = random.sample(possible_tasks, k)
    return random_tasks



def lpt_asp(tasks: List[Task], action_mask: np.array) -> int:
    """
    LPT: longest processing time first. Determines the unfinished task with the longest processing time for ASP
    :param tasks:
    :param action_mask:
    :return:
    """
    possible_tasks = get_active_task_dict_asp(tasks)
    task_index = -1
    longest_processing_time = 0
    for i, task in enumerate(tasks):
        if not task.done and task.task_index in possible_tasks.keys():
            if task.max_execution_times_setup > longest_processing_time:
                longest_processing_time = task.max_execution_times_setup
                task_index = i
    return task_index

def lpt_asp_first_k(tasks: List[Task], k: int) -> List[int]:
    """
    LPT: longest processing time first. Determines the first k unfinished tasks with the longest processing times for ASP.

    :param tasks: List of task objects, so one instance.
    :param k: Number of tasks to select with the longest processing times.

    :return: List of indices of the tasks selected according to the heuristic.
    """
    possible_tasks = get_active_task_dict_asp(tasks)
    task_runtimes = []

    # Collect tasks with their runtimes if they are possible and not done
    for i, task in enumerate(tasks):
        if not task.done and task.task_index in possible_tasks.keys():
            task_runtimes.append((i, task.max_execution_times_setup))

    # Sort tasks by runtime in descending order and select the first k
    task_runtimes.sort(key=lambda x: x[1], reverse=True)
    selected_tasks = [task[0] for task in task_runtimes[:k]]

    return selected_tasks

def get_operation_time_per_tasks(tasks: List[Task]) -> List[int]:
    """
    Helper function to get the operation time per task

    :param tasks: List of task objects, so one instance

    :return: List of operation times per task
    """

    operation_time_per_tasks = [0] * len(tasks)

    #  machine bottleneck feature = ”number of unscheduled operations per machine” or ”total duration (sum of processing times) of unscheduled operations per machine
    #  mapping of many machines are used dynamically
    machines_counter_dynamic = [0] * len(tasks[0].machines)
    for task in tasks:
        if not task.done:
            for index in range(len(task.machines)):
                if task.machines[index] == 1:
                    machines_counter_dynamic[index] += 1

    for i, task in enumerate(tasks):
        if task.done:
            operation_time_per_tasks[task.task_index] = task.finished - task.started
        else:
            weight_up = 0
            weight_down = 0
            for index in range(len(task.machines)):
                if task.machines[index] == 1:
                    weight_up += machines_counter_dynamic[index] * task.execution_times[index]
                    weight_down += machines_counter_dynamic[index]
            weighted_average_runtime = weight_up / weight_down

            operation_time_per_tasks[task.task_index] = weighted_average_runtime

    return operation_time_per_tasks

def compute_lrm_srm_lwkr_mwkr(tasks: List[Task], should_take_max, should_include_task_itself):
    possible_tasks = get_active_task_dict_asp(tasks)

    operation_time_per_tasks = get_operation_time_per_tasks(tasks)
    estimated_remaining_processing_time_per_task = [0] * len(tasks)

    # estimated_remaining_processing_time_per_task
    longest_processing_time = 0
    shortest_processing_time = np.inf


    task_index = 0
    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys():
            # estimated_remaining_processing_time_per_task[task.task_index] = operation_time_per_tasks[task.task_index]
            task_successor_index = task.parent_index

            while task_successor_index is not None:
                estimated_remaining_processing_time_per_task[task.task_index] += operation_time_per_tasks[task_successor_index]
                task_successor_index = tasks[task_successor_index].parent_index

            if should_include_task_itself:
                estimated_remaining_processing_time_per_task[task.task_index] += operation_time_per_tasks[task.task_index]

            # find the task with the maximum estimated remaining processing time
            if should_take_max:
                if estimated_remaining_processing_time_per_task[task.task_index] > longest_processing_time:
                    longest_processing_time = estimated_remaining_processing_time_per_task[task.task_index]
                    task_index = i
            else:
                if estimated_remaining_processing_time_per_task[task.task_index] < shortest_processing_time:
                    shortest_processing_time = estimated_remaining_processing_time_per_task[task.task_index]
                    task_index = i

    return task_index

def compute_lrm_srm_lwkr_mwkr_first_k(tasks: List[Task], should_take_max: bool, should_include_task_itself: bool, k: int) -> List[int]:
    possible_tasks = get_active_task_dict_asp(tasks)

    operation_time_per_tasks = get_operation_time_per_tasks(tasks)
    estimated_remaining_processing_time_per_task = [0] * len(tasks)
    task_processing_times = []

    # Calculate the estimated remaining processing time for each task
    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys():
            task_successor_index = task.parent_index

            while task_successor_index is not None:
                estimated_remaining_processing_time_per_task[task.task_index] += operation_time_per_tasks[task_successor_index]
                task_successor_index = tasks[task_successor_index].parent_index

            if should_include_task_itself:
                estimated_remaining_processing_time_per_task[task.task_index] += operation_time_per_tasks[task.task_index]

            task_processing_times.append((i, estimated_remaining_processing_time_per_task[task.task_index]))

    # Sort tasks by remaining processing time
    task_processing_times.sort(key=lambda x: x[1], reverse=should_take_max)

    # Select the first k tasks
    selected_tasks = [task[0] for task in task_processing_times[:k]]

    return selected_tasks

def lrm_asp(tasks: List[Task], action_mask: np.array) -> int:
    """
    LRM (Longest Remaining Processing Time) =
    the maximum sum of the processing times of the operations located on the branch
    leading to the root (excluding the execution time of the operation itself).
    :param tasks:
    :param action_mask:
    :return:
    """
    task_index = compute_lrm_srm_lwkr_mwkr(tasks, should_take_max = True, should_include_task_itself = False )

    return task_index

def lrm_asp_first_k(tasks: List[Task], k: int):
    """
    LRM (Longest Remaining Processing Time) =
    first k operations for the maximum sum of the processing times of the operations located on the branch
    leading to the root (excluding the execution time of the operation itself).
    :param k:
    :param tasks:
    :return:
    """
    task_indexes = compute_lrm_srm_lwkr_mwkr_first_k(tasks, should_take_max = True, should_include_task_itself = False, k = k )

    return task_indexes


def srm_asp(tasks: List[Task], action_mask: np.array) -> int:
    """
    SRM (Shortest Remaining Processing Time) =
    the shortest sum of the processing times of the operations located on the branch
    leading to the root (excluding the execution time of the operation itself).
    :param tasks:
    :param action_mask:
    :return:
    """

    task_index = compute_lrm_srm_lwkr_mwkr(tasks, should_take_max = False, should_include_task_itself = False )

    return task_index


def srm_asp_first_k(tasks: List[Task], k: int):
    """
    SRM (Shortest Remaining Processing Time) =
    first k operations for  the shortest sum of the processing times of the operations located on the branch
    leading to the root (excluding the execution time of the operation itself).
    :param k:
    :param tasks:
    :return:
    """

    task_indexes = compute_lrm_srm_lwkr_mwkr_first_k(tasks, should_take_max = False, should_include_task_itself = False, k = k )

    return task_indexes



def lwkr_asp(tasks: List[Task], action_mask: np.array) -> int:
    """
    LWKR (Least Work Remaining) = the minimum sum of the times of the operations located on the branch leading to the root (including the execution time of the operation itself).
    :param tasks:
    :param action_mask:
    :return:
    """
    task_index = compute_lrm_srm_lwkr_mwkr(tasks, should_take_max = False, should_include_task_itself = True )

    return task_index

def lwkr_asp_first_k(tasks: List[Task], k: int):
    """
    LWKR (Least Work Remaining) = first k operations for  the minimum sum of the times of the operations located on the branch leading to the root (including the execution time of the operation itself).
    :param k:
    :param tasks:
    :return:
    """
    task_indexes = compute_lrm_srm_lwkr_mwkr_first_k(tasks, should_take_max = False, should_include_task_itself = True, k = k )

    return task_indexes

def mwkr_asp(tasks: List[Task], action_mask: np.array) -> int:
    """
    MWKR (Most Work Remaining) = the maximum sum of the times of the operations located on the branch leading to the root (including the execution time of the operation itself).
    :param tasks:
    :param action_mask:
    :return:
    """
    task_index = compute_lrm_srm_lwkr_mwkr(tasks, should_take_max = True, should_include_task_itself = True )

    return task_index

def mwkr_asp_first_k(tasks: List[Task], k: int):
    """
    MWKR (Most Work Remaining) =  first k operations the maximum sum of the times of the operations located on the branch leading to the root (including the execution time of the operation itself).
    :param k:
    :param tasks:
    :return:
    """
    task_indexes = compute_lrm_srm_lwkr_mwkr_first_k(tasks, should_take_max = True, should_include_task_itself = True, k= k )

    return task_indexes

def edd(tasks: List[Task], action_mask: np.array) -> int:
    """
    EDD: earliest due date. Determines the job with the smallest deadline

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    if np.sum(action_mask) == 1:
        chosen_job = np.argmax(action_mask)
    else:
        num_jobs = action_mask.shape[0] - 1
        num_tasks_per_job = len(tasks) / num_jobs
        deadlines = np.full(num_jobs + 1, np.inf)

        for job_i in range(num_jobs):
            idx = int(num_tasks_per_job * job_i)
            deadlines[job_i] = tasks[idx].deadline

        deadlines = np.where(action_mask == 1, deadlines, np.full(deadlines.shape, np.inf))
        chosen_job = np.argmin(deadlines)
    return chosen_job

def spt(tasks: List[Task], action_mask: np.array) -> int:
    """
    SPT: shortest processing time first. Determines the job of which the next unfinished task has the lowest runtime

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    if np.sum(action_mask) == 1:
        chosen_job = np.argmax(action_mask)
    else:
        num_jobs = action_mask.shape[0] - 1
        runtimes = np.full(num_jobs + 1, np.inf)
        active_task_dict = get_active_task_dict(tasks)

        for i in range(num_jobs):
            if i in active_task_dict.keys():
                task_idx = active_task_dict[i]
                runtimes[i] = tasks[task_idx].runtime
        runtimes = np.where(action_mask == 1, runtimes, np.full(runtimes.shape, np.inf))
        chosen_job = np.argmin(runtimes)
    return chosen_job


def mtr(tasks: List[Task], action_mask: np.array) -> int:
    """
    MTR: most tasks remaining. Determines the job with the least completed tasks

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    if np.sum(action_mask) == 1:
        chosen_job = np.argmax(action_mask)
    else:
        tasks_done = np.zeros(len(tasks) + 1)
        possible_tasks = get_active_task_dict(tasks)
        for _, task in enumerate(tasks):
            if task.done and task.job_index in possible_tasks.keys():
                tasks_done[possible_tasks[task.job_index]] += 1

        task_mask = np.zeros(len(tasks) + 1)
        for job_id, task_id in possible_tasks.items():
            if action_mask[job_id] == 1:
                task_mask[task_id] += 1
        tasks_done = np.where(task_mask == 1, tasks_done, np.full(tasks_done.shape, np.inf))
        tasks_done[-1] = np.inf
        chosen_task = np.argmin(tasks_done)
        chosen_job = tasks[chosen_task].job_index
    return chosen_job


def ltr(tasks: List[Task], action_mask: np.array) -> int:
    """
    LTR: least tasks remaining. Determines the job with the most completed tasks

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    if np.sum(action_mask) == 1:
        chosen_job = np.argmax(action_mask)
    else:
        tasks_done = np.zeros(len(tasks) + 1)
        possible_tasks = get_active_task_dict(tasks)
        for _, task in enumerate(tasks):
            if task.done and task.job_index in possible_tasks.keys():
                tasks_done[possible_tasks[task.job_index]] += 1
        task_mask = np.zeros(len(tasks) + 1)
        for job_id, task_id in possible_tasks.items():
            if action_mask[job_id] == 1:
                task_mask[task_id] += 1
        tasks_done = np.where(task_mask == 1, tasks_done, np.full(tasks_done.shape, -1))
        tasks_done[-1] = -1
        chosen_task = np.argmax(tasks_done)
        chosen_job = tasks[chosen_task].job_index
    return chosen_job


def random_task(tasks: List[Task], action_mask: np.array) -> int:
    """
    Returns a random task

    :param tasks: Not needed
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """

    chosen_job = None
    if np.sum(action_mask) == 1:
        chosen_job = np.argmax(action_mask)
    else:
        valid_values_0 = np.where(action_mask > 0)[0]

        if len(valid_values_0) > 2:
            chosen_job = np.random.choice(valid_values_0, size=1)[0]
        elif len(valid_values_0) == 0:
            print('this is not possible')
        else:
            chosen_job = np.random.choice(valid_values_0, size=1)[0]
    return chosen_job


def choose_random_machine(chosen_task, machine_mask) -> int:
    """
    Determines a random machine which is available according to the mask and chosen task. Useful for the FJSSP.

    :param chosen_task: ID of the task that is scheduled on the selected machine
    :param machine_mask: Machine mask from the environment that is to receive the machine action chosen by this function

    :return: Index of the chosen machine

    """
    machine_mask = np.array(np.where(machine_mask > 0))
    idx_valid_machine = np.where(machine_mask[0] == chosen_task)
    valid_machines = machine_mask[1][idx_valid_machine]
    chosen_machine = np.random.choice(valid_machines, size=1)[0]
    return chosen_machine


def choose_first_machine(chosen_task, machine_mask) -> int:
    """
    Determines the first (by index) machine which is available according to the mask and chosen task. Useful for the
    FJSSP

    :param chosen_task: ID of the task that is scheduled on the selected machine
    :param machine_mask: Machine mask from the environment that is to receive the machine action chosen by this function

    :return: Index of the chosen machine

    """
    machine_mask = np.array(np.where(machine_mask > 0))
    idx_valid_machine = np.where(machine_mask[0] == chosen_task)
    valid_machines = machine_mask[1][idx_valid_machine]
    return valid_machines[0]

class HeuristicSelectionAgent:
    """
    This class can be used to get the next task according to the heuristic passed as string abbreviation (e.g. EDD).
    If you want to edit a shortcut, or add one for your custom heuristic, adapt/extend the task_selection dict.

    :Example:

    .. code-block:: python

        def my_custom_heuristic():
            ...<function body>...

    or

    .. code-block:: python

        self.task_selections = {
            'rand': random_task,
            'XYZ': my_custom_heuristic
            }

    """

    def __init__(self) -> None:

        super().__init__()
        # Map heuristic ids to corresponding function
        self.task_selections = {
            'rand': random_task,
            'EDD': edd,
            'SPT': spt,
            'MTR': mtr,
            'LTR': ltr,
            'EDD_ASP': edd_asp,
            'SPT_ASP': spt_asp,
            'MPO_ASP': mpo_asp,
            'LPO_ASP': lpo_asp,
            # 'MPO_ASP_STANDARD': mpo_asp_standard,
            # 'LPO_ASP_STANDARD': lpo_asp_standard,
            'LPO_ROOT_ASP': lpo_root_asp,
            'MPO_ROOT_ASP': mpo_root_asp,
            'rand_asp': random_task_asp,
            'LPT_ASP': lpt_asp,
            'LRM_ASP': lrm_asp,
            'SRM_ASP': srm_asp,
            'LWKR_ASP': lwkr_asp,
            'MWKR_ASP': mwkr_asp,
            'LETSA': letsa,
            'ECT_ASP': ect_asp
        }

        self.task_selections_first_k = {
            'SPT_ASP_FIRST_K': spt_asp_first_k,
            'LPT_ASP_FIRST_K': lpt_asp_first_k,
            'EDD_ASP_FIRST_K': edd_asp_first_k,
            'MPO_ASP_FIRST_K': mpo_asp_first_k,
            'LPO_ASP_FIRST_K': lpo_asp_first_k,
            'LRM_ASP_FIRST_K': lrm_asp_first_k,
            'SRM_ASP_FIRST_K': srm_asp_first_k,
            'LWKR_ASP_FIRST_K': lwkr_asp_first_k,
            'MWKR_ASP_FIRST_K': mwkr_asp_first_k,
            'rand_asp_first_k': random_task_asp_first_k,
            'LPO_ROOT_ASP_FIRST_K': lpo_root_asp_first_k,
            'MPO_ROOT_ASP_FIRST_K': mpo_root_asp_first_k,
            'ECT_ASP_FIRST_K': ect_asp_first_k,
        }


    def __call__(self, tasks: List, action_mask: np.array, task_selection: str, feasible_tasks = None, visited = None, max_deadline = None, k = None, ends_of_machine_occupancies = None):
        """
        Selects the next heuristic function according to the heuristic passed as string abbreviation
        and the assignment in the task_selections dictionary

        :param tasks: List of task objects, so one instance
        :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic.
        :param task_selection: Heuristic string abbreviation (e.g. EDD)

        :return: Index of the job or task selected according to the heuristic

        """

        if task_selection == 'ECT_ASP_FIRST_K':
            choose_tasks = self.task_selections_first_k[task_selection]
            chosen_tasks = choose_tasks(tasks, ends_of_machine_occupancies=ends_of_machine_occupancies, k=k)
            return chosen_tasks
        elif task_selection in self.task_selections_first_k.keys():
            choose_tasks = self.task_selections_first_k[task_selection]
            chosen_tasks = choose_tasks(tasks, k=k)
            return chosen_tasks
        elif task_selection == 'ECT_ASP':
            choose_task = self.task_selections[task_selection]
            chosen_task, completion_time = choose_task(tasks, ends_of_machine_occupancies=ends_of_machine_occupancies)
            return chosen_task, completion_time
        elif task_selection == 'LETSA':
            choose_task = self.task_selections[task_selection]
            chosen_task, completion_time = choose_task(tasks, action_mask, feasible_tasks, visited, max_deadline)
            return chosen_task, completion_time
        else:
            choose_task = self.task_selections[task_selection]
            chosen_task = choose_task(tasks, action_mask)
            return chosen_task
