
import os
import ast
import argparse


def extract_makespan_mean(file_path):
    with open(file_path, 'r') as file:
        last_line = file.readlines()[-1]
        data = ast.literal_eval(last_line)
        if 'agent' in data and 'makespan_mean' in data['agent']:
            return data['agent']['makespan_mean']
    return float('inf')

def find_file_with_smallest_makespan(directory):
    smallest_makespan = float('inf')
    smallest_file = None

    for root, _, files in os.walk(directory):
        for file in files:
            if 'test' in file:
                file_path = os.path.join(root, file)
                makespan_mean = extract_makespan_mean(file_path)
                if makespan_mean < smallest_makespan:
                    smallest_makespan = makespan_mean
                    smallest_file = file_path

    return smallest_file, smallest_makespan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find the file with the smallest makespan_mean.')
    parser.add_argument('directory', type=str, help='Directory to search for files')
    args = parser.parse_args()

    smallest_file, smallest_makespan = find_file_with_smallest_makespan(args.directory)

    if smallest_file:
        print(f"The file with the smallest makespan_mean is: {smallest_file} with a makespan_mean of {smallest_makespan}")
    else:
        print("No file containing 'test' found in the directory.")
