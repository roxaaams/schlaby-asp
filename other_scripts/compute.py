import argparse
from src.environments.environment_loader import EnvironmentLoader
from src.agents.train_test_utility_functions import get_agent_class_from_config, load_config, load_data

def compute(env_config, data):
     for test_i in range(len(data)):
        env, _ = EnvironmentLoader.load(env_config, data=[data[test_i]], binary_features=None)

        no_op = len(env.tasks)

        mean_execution_times = 0
        count_mean_execution_times = 0

        mean_no_ma_op = 0
        # we don't exactly how many machines are since the binary list is anyway a bit larger, so we need to get the max index from the binary list that is a 1
        max_ma_index = 0

        max_height = 0
        visited_for_height = dict()

        mean_branching = 0
        count_branching = 0

        for task in env.tasks:
            count_mean_no_ma_op = 0
            for i in range(len(task.machines)):
                if task.machines[i] == 1:
                    mean_execution_times += task.execution_times[i]
                    count_mean_execution_times += 1
                    count_mean_no_ma_op += 1
                    max_ma_index = max(max_ma_index, i)
            mean_no_ma_op += count_mean_no_ma_op

            if not task.task_index in visited_for_height:
                current_height = 1
                task_successor_index = task.parent_index
                visited_for_height[task.task_index] = 1

                while task_successor_index is not None:
                    visited_for_height[task_successor_index] = 1
                    current_height += 1
                    task_successor_index = env.tasks[task_successor_index].parent_index
                #   4. tree height
                max_height = max(max_height, current_height)
            if len(task.children) > 0:
                mean_branching += len(task.children)
                count_branching += 1



         #     2.a the average processing time of operations on machines
        mean_execution_times /= count_mean_execution_times

         #     3.a The average nr machines/operation
        mean_no_ma_op /= max_ma_index

        # 5.a mean for branching (no children)
        mean_branching /= count_branching


        variance_execution_times = 0
        variance_no_ma_op = 0

        variance_branching = 0

        for task in env.tasks:
            count_mean_no_ma_op = 0
            for index in range(len(task.machines)):
                if task.machines[index] == 1:
                    variance_execution_times += (task.execution_times[index] - mean_execution_times) ** 2
                    count_mean_no_ma_op += 1
            variance_no_ma_op += (count_mean_no_ma_op - mean_no_ma_op) ** 2


            if len(task.children) > 0:
               variance_branching += (len(task.children) -  mean_branching) ** 2

         #  2.b the standard deviation of processing time of operations on machines
        variance_execution_times /= count_mean_execution_times
        variance_execution_times = variance_execution_times ** 0.5

         # 3.b The standard deviation of nr machines/operation
        variance_no_ma_op /= max_ma_index
        variance_no_ma_op = variance_no_ma_op ** 0.5

        # 5.b mean for branching (no children)
        variance_branching /= count_branching
        variance_branching = variance_branching ** 0.5

        print(no_op, mean_execution_times, variance_execution_times, mean_no_ma_op, variance_no_ma_op, max_height, mean_branching, variance_branching)


def get_perser_args():
    # Arguments for function
    parser = argparse.ArgumentParser(description='Compute means and standard deviations')

    parser.add_argument('-fp', '--config_file_path', type=str, required=True,
                        help='Path to config file you want to use for training')

    args = parser.parse_args()

    return args


def main(external_config=None):

    # get config_file and binary_features from terminal input
    parse_args = get_perser_args()
    config_file_path = parse_args.config_file_path

    # get config and data
    config = load_config(config_file_path, external_config)
    data = load_data(config)

    compute(config, data)


if __name__ == '__main__':

    main()
