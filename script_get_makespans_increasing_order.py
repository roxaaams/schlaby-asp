
import os
import ast
import argparse
import re



def extract_makespan_mean(file_path):
    with open(file_path, 'r') as file:
        last_line = file.readlines()[-1]
        data = ast.literal_eval(last_line)
        if 'agent' in data and 'makespan_mean' in data['agent']:
            return data['agent']['makespan_mean']
    return float('inf')

def get_sorted_makespans(directory):
    makespan_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if 'test' in file:
                file_path = os.path.join(root, file)
                makespan_mean = extract_makespan_mean(file_path)
                makespan_list.append((makespan_mean, file))
    sorted_makespan_list = sorted(makespan_list, key=lambda x: x[0])

    return sorted_makespan_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find the file with the smallest makespan_mean.')
    parser.add_argument('directory', type=str, help='Directory to search for files')
    args = parser.parse_args()

    digit_dict = {i: 0 for i in range(0, 10)}

    makespan_list = get_sorted_makespans(args.directory)
    smallest_makespan = makespan_list[0][0]
    for makespan, file_name in makespan_list:  
        match = re.search(r'_(\d+)\.txt', file_name)
        if (match):
            numeric_part = match.group(1)
            # Update digit_dict based on numeric_part
            for i in range(0, 10):
                if numeric_part[i] == '1':
                    digit_dict[i] = digit_dict[i] + 1
        # print(f"File: {file_name}, Makespan Mean: {makespan}")

    print("Digit Dictionary:", digit_dict)

