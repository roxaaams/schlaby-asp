from boltons.setutils import IndexedSet

class SetQueue:
    def __init__(self):
        self.queue = IndexedSet()

    def put(self, item):
        self.queue.add(item)

    def get(self):
        return self.queue.pop()

    def empty(self):
        return len(self.queue) == 0

    def contain(self, element):
        return element in self.queue
