import os.path
import numpy as np
import pandas as pd
import scipy.io
import common.utils.time as time
from common.Parser import Parser


class SeizureParser(Parser):
    """
    Parse seizure data to desired format for training and prediction
    """
    def __init__(self, data_dir, target, pipeline):
        self.data_dir = data_dir
        self.target = target
        self.pipeline = pipeline
        self.labels = pd.read_csv(self.data_dir + '/train_and_test_data_labels_safe.csv')

    # generator to iterate over competition mat data
    def load_data(self, response):
        done = False
        i = 0
        sequence_id = -1
        while not done:
            i += 1
            if response != '':
                filename = '%s/%s_%d%s.mat' % (self.data_dir, self.target, i, response)
            else:
                filename = '%s/new_%s_%d%s.mat' % (self.data_dir, self.target, i, response)
            # print filename
            if os.path.exists(filename):
                data = scipy.io.loadmat(filename, verify_compressed_data_integrity=False)
                d = data['dataStruct']['data'][0][0].transpose()
                if 'sequence' in data['dataStruct'].dtype.names:
                    sequence = int(data['dataStruct']['sequence'][0][0][0][0])
                    if i % 6 == 1:
                        if sequence != 1:
                            print 'check file %s for errror in sequence', filename
                        sequence_id += 1
                if response != '':
                    file_id = '%s_%d%s.mat' % (self.target, i, response)
                else:
                    file_id = 'new_%s_%d%s.mat' % (self.target, i, response)
                if file_id in self.labels.image[self.labels.safe==0].tolist():
                    continue
                yield ({'data': d, 'file_id': file_id, 'sequence_id': sequence_id})
            else:
                if i == 1:
                    raise Exception("file %s not found" % filename)
                # if response == '_1':
                #     j = 0
                #     while not done:
                #         j += 1
                #         filename = '%s/%s_%d.mat' % (self.data_dir, self.target, j)
                #         file_id = '%s_%d.mat' % (self.target, j)
                #         if os.path.exists(filename):
                #             if file_id in self.labels.image.tolist():
                #                 data = scipy.io.loadmat(filename, verify_compressed_data_integrity=False)
                #                 d = data['dataStruct']['data'][0][0].transpose()
                #                 sequence_id += 1
                #                 yield ({'data': d, 'file_id': file_id, 'sequence_id': sequence_id})
                #         else:
                #             done = True
                done = True

    def parse(self, data_type):
        response = {'ictal': '_1', 'interictal': '_0', 'test': ''}
        Y = {'ictal': 1, 'interictal': 0, 'test': None}
        start = time.get_seconds()
        print 'Loading data',
        mat_data = self.load_mat_data(response[data_type])
        print '(%ds)' % (time.get_seconds() - start)
        # for each data point in ictal, interictal and test,
        # generate (X, <y>, <latency>) per channel

        data = self.process_raw_data(mat_data, Y[data_type])

        if len(data) == 4:
            X, y, file_id, sequence_id = data
            return {
                'X': X,
                'y': y,
                'sequence_id': sequence_id
            }
        else:
            X, file_id = data
            return {
                'X': X,
                'File': ','.join(file_id)
            }

    def process_mat_data(self, mat_data, y_value):
        start = time.get_seconds()

        print 'Transforming data',
        X = []
        y = []
        file_id = []
        sequence_id = []

        for segment in mat_data:
            if np.sum(np.sum(segment['data'], axis=1)) == 0:
                continue
            transformed_data = self.pipeline.apply(segment['data'])
            X.append(transformed_data)
            if y_value is not None:
                y.append(y_value)
            file_id.append(segment['file_id'])
            sequence_id.append(segment['sequence_id'])

        print '(%ds)' % (time.get_seconds() - start)

        X = np.array(X)
        y = np.array(y) if len(y) > 0 else None

        if y_value is None:
            print 'X', X.shape
            return X, file_id
        else:
            print 'X', X.shape, 'y', y.shape
            return X, y, file_id, sequence_id


class Writer(object):
    def __init__(self, submission_dir, submission_id):
        self.file_name = '%s/submission_%s.csv' % (submission_dir, submission_id)

    def write(self, data):
        if isinstance(data, pd.DataFrame):
            data.to_csv(self.file_name, index=False)
        else:
            "Writer failed due to output type"

