class Machine:
    def __init__(self):
        self.intervals = []

    def add_interval(self, index, task):
        if index == -1:
            self.intervals.append(task)
        else:
            self.intervals.insert(index, task)

    def add_last_interval(self, task):
        self.intervals.append(task)

    def get_duration(self, task):
        return self.tasks[task]

    def get_tasks_len(self):
        return len(self.tasks)

    def get_int_len(self):
        return len(self.intervals)

    def get_last_int(self):
        return self.intervals[-1]

    def get_before_last_int(self):
        return self.intervals[-2]

    def get_int(self, index):
        return self.intervals[index]

    def __str__(self):
        return 'id: {self.id}, intervals: {self.intervals}, '.format(self=self)
