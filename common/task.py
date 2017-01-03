from collections import namedtuple

TaskCore = namedtuple('TaskCore', ['cached_data_loader', 'data_dir', 'target', 'pipeline', 'parser',
                                   'classifier', 'cv_ratio', 'train', 'test'])


class Task(object):
    """
    A Task computes some work and outputs a dictionary which will be cached on disk.
    If the work has been computed before and is present in the cache, the data will
    simply be loaded from disk and will not be pre-computed.
    """
    def __init__(self, task_core):
        self.task_core = task_core

    def filename(self):
        raise NotImplementedError("Implement this")

    def run(self):
        return self.task_core.cached_data_loader.load(self.filename(), self.load_data)