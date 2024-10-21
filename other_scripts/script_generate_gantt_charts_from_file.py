import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

def read_first_n_lines(file_path, n):
    with open(file_path, 'r') as file:
        lines = [file.readline().strip() for _ in range(n)]
    return lines

def parse_line(line):
    name, tasks_str = line.split(':')
    tasks = tasks_str.strip().split(' ')
    parsed_tasks = []
    for i in range(0, len(tasks), 4):
        task_id = int(tasks[i])
        machine = int(tasks[i+1])
        start = int(tasks[i+2])
        finish = int(tasks[i+3])
        parsed_tasks.append((task_id, machine, start, finish))
    return name.strip(), parsed_tasks

def create_gantt_chart(data, title, output_file):
    df = pd.DataFrame(data, columns=['Task', 'Machine', 'Start', 'Finish'])
    fig, ax = plt.subplots(figsize=(10, 6))

    unique_tasks = df['Task'].unique()
    colors = plt.cm.get_cmap('tab20', len(unique_tasks))
    legend_patches = []

    for i, task in enumerate(unique_tasks):
        task_data = df[df['Task'] == task]
        color = colors(i)
        legend_patches.append(mpatches.Patch(color=color, label=f"Op. {task}"))
        ax.broken_barh(task_data[['Start', 'Finish']].values - task_data['Start'].values[:, None],
                       (task_data['Machine'].values[0] - 0.4, 0.8), facecolors=color)
        for _, row in task_data.iterrows():
            ax.text(row['Start'] + (row['Finish'] - row['Start']) / 2, row['Machine'], '',
                    va='center', ha='center', color='white', fontsize=12, fontweight='bold')

    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_yticks(df['Machine'].unique())
    ax.set_yticklabels(df['Machine'].unique())
    ax.grid(True)
    ax.set_title(title)
    ax.legend(handles=legend_patches, title="Legend")
    # plt.show()
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    file_path = '../best_makespan_asp_tubes.txt'
    lines = read_first_n_lines(file_path, 8)


    for line in lines:
        min_start = float('inf')
        max_end = float('-inf')
        name, tasks = parse_line(line)
        output_file = f"gantts/{name}_gantt_chart.png"


        for i in range(len(tasks)):
            min_start = min(min_start, tasks[i][2])
            max_end = max(max_end, tasks[i][3])

        makespan = max_end - min_start
        print(f"makespan for {name}: {makespan}")

        create_gantt_chart(tasks, f"Schedule for {name}", output_file)
