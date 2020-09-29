# from collections import namedtuple
# Instance = namedtuple("Instance", 'task data label')
# Instance = namedtuple("Instance", ['task', 'data', 'label'])


class Instance(object):
    def __init__(self, task, data, label):
        self.task = task
        self.data = data
        self.label = label

    def __str__(self):
        return f"Task: {self.task}, Data item: {self.data}, Label: {self.label}"

    def __repr__(self):
        return f"Task: {self.task}, Data item: {self.data}, Label: {self.label}"

