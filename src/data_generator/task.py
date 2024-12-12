"""This file provides the Task class."""
# Standard package import
from typing import List
from typing import Dict

class Task:
    """
    This class can be used to model tasks of a scheduling problem.
    Multiple tasks can be used to model jobs of a scheduling problem.

    :param job_index: index of the job to which multiple tasks belong
    :param task_index: index of the task within the job (unique within job and gives order of tasks)
    :param machines: list of machine indices applicable for the specific task (alternatives - pick one)
    :param tools: list of tool indices applicable for the specific task (necessary - pick all)
    :param deadline: time of deadline for the task
    :param instance_hash: Individual hash to represent the instance
    :param done: bool to determine status as done
    :param runtime: time left for the task
    :param started: time the task started at in the schedule
    :param finished: time the task finished at in the schedule
    :param selected_machine: selected machine index from the machines list for the specific schedule
    :param _n_machines: number of all available machines in the scheduling problem instance
    :param _n_tools: number of all available tools in the scheduling problem instance
    :param _feasible_machine_from_instance_init: index of machine in the given instance generated by initial environment
        run to generate deadline time
    :param _feasible_order_index_from_instance_init: index of task in the given instance
        generated by initial environment run to generate deadline time
    :param children: list of predecessor indexes of current task
    :param parent_index: parent index of current task
    :param quantity: quantity of task to be produced
    :param machines_indexes: indexes of machines
    :param execution_times:
    :param setup_times:
    :param average_runtime:
    """
    def __init__(self, job_index: int,
                 task_index: int = None, machines: List[int] = None,
                 tools: List[int] = None, deadline: int = None, instance_hash: int = None, done: bool = None,
                 runtime: int = None, started: int = None, finished: int = None, selected_machine: int = None,
                 _n_machines: int = None, _n_tools: int = None, _feasible_machine_from_instance_init: int = None,
                 _feasible_order_index_from_instance_init: int = None,
                 children: List[int] = None,
                 parent_index = None,
                 quantity: int = 1,
                 execution_times: Dict[int, int] = None,
                 setup_times: Dict[int, int] = None,
                 setup_time: int = None,
                 should_multiply_quantity_to_execution_times = False,
                 task_id = None,
                 filename = None,
                 deleted = False,
                 average_runtime = None):

        # test for correct data type of required and throw type error otherwise
        if not isinstance(job_index, int) or not isinstance(task_index, int):
            raise TypeError("Job index and task index must be of type int.")

        # public - static - required - don't touch after init
        self.job_index = job_index
        self.task_index = task_index

        # public - static - optional - don't touch after init
        self.machines = machines
        self.tools = tools
        self.deadline = deadline
        self.instance_hash = instance_hash

        # public - non-static - optional
        self.done = done
        self.runtime = runtime
        self.started = started
        self.finished = finished
        self.selected_machine = selected_machine

        # protected - optional
        self._n_machines = _n_machines
        self._n_tools = _n_tools
        self._feasible_machine_from_instance_init = _feasible_machine_from_instance_init
        self._feasible_order_index_from_instance_init = _feasible_order_index_from_instance_init

        # public - non-static
        self.task_id = task_id
        self.filename = filename
        self.children = children
        self.parent_index = parent_index
        self.quantity = quantity
        self.setup_times = setup_times
        self.setup_time = setup_time
        self.execution_times = execution_times
        if should_multiply_quantity_to_execution_times == True:
            self.max_execution_times_setup = 0
            self.average_execution_times_setup = 0
            for machine_id in self.execution_times:
                self.execution_times[machine_id] *= self.quantity
                if self.execution_times[machine_id]+self.setup_times[machine_id]>self.max_execution_times_setup:
                    self.max_execution_times_setup = self.execution_times[machine_id]+self.setup_times[machine_id]
                self.average_execution_times_setup += self.execution_times[machine_id]+self.setup_times[machine_id]
            self.runtime = int(runtime * quantity)  # max execution time (without setup)
            self.average_execution_times_setup /= len(self.execution_times)


    def __str__(self) -> str:
        return f"Task - job index {self.job_index} - task index {self.task_index} - parent_index {self.parent_index} - children {self.children}"

    def str_info(self) -> str:
        return f"Job index {self.job_index}\nTask index {self.task_index}\n "

    def str_routine_info(self) -> str:
        machines_str = ""
        for i in range(len(self.machines)):
            if self.machines[i] == 1:
                machines_str += f"Mach. {i}: {self.execution_times[i]} "
        return f"{self.task_index} & {self.parent_index} &  {machines_str}"

    def str_schedule_info(self) -> str:
        return f"Task id from BOM {self.task_id}, Task index in code {self.task_index}, Selected Machine {str(self.selected_machine)}, Start {str(self.started)}, Finish {str(self.finished)}"

    def str_schedule_info_short(self) -> str:
        return f"{self.task_index} {str(self.selected_machine)} {str(self.started)} {str(self.finished)}"

    def str_setup_info(self) -> str:
        return f"Task_id: {self.task_id}, Task index {self.task_index}, Setup time {str(self.setup_time)}, Average execution and setup time {str(self.average_execution_times_setup)}, Max execution and setup time {str(self.max_execution_times_setup)}"

