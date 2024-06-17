import subprocess
from itertools import product
import datetime

def execute_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")


def generate_and_process_binary_masks():
    prefix_train_cmd = "python -m src.agents.train -fp training/ppo/config_ASP_WIDE.yaml -bf"
    prefix_test_cmd = "python -m src.agents.test -fp testing/ppo/config_ASP_WIDE.yaml -bf"

    results_dir = "results/asp_wide/"

    upper_bound = 4
    lower_bound = 2

    null_mask = '0000000000'

    counter = 0
    run_heuristics = 1 # should run the heuristics and print their results only once if =1
    for mask_tuple in product([0, 1], repeat=10):
        mask = ''.join(map(str, mask_tuple))
        if mask != null_mask:
            print(mask)
            if counter >= lower_bound and counter <= upper_bound:
                train_cmd = f"{prefix_train_cmd} {mask} > {results_dir}train_{counter}_{mask}.txt"
                test_cmd = f"{prefix_test_cmd} {mask} -rh {run_heuristics} > {results_dir}test_{counter}_{mask}.txt"

                first_time = datetime.datetime.now()
                execute_cmd(train_cmd)
                later_time = datetime.datetime.now()
                print("train time: ", later_time - first_time)

                first_time = datetime.datetime.now()
                execute_cmd(test_cmd)
                later_time = datetime.datetime.now()
                print("test time: ", later_time - first_time)

                run_heuristics = 0
            elif counter > upper_bound:
                return
        counter += 1

# Example usage:
generate_and_process_binary_masks()
