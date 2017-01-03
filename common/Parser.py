class Parser(object):
    """
    Parse seizure data to desired format for training and prediction
    """
    def __init__(self, data_dir, target, pipeline):
        self.data_dir = data_dir
        self.target = target
        self.pipeline = pipeline

    # generator to iterate over competition mat data
    def load_data(self, response):
        raise NotImplementedError("Implement this")

    def parse(self, data_type):
        raise NotImplementedError("Implement this")

    def process_mat_data(self, mat_data, y_value):
        raise NotImplementedError("Implement this")