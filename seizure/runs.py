import gc
import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import fmin_cobyla
from sklearn.metrics import roc_auc_score
from common.task import TaskCore
from common.data import CachedDataLoader, makedirs, JSDict
from common.utils import time
from seizure.io import Writer, SeizureParser
from seizure.scores import get_score_summary, print_results
from seizure.tasks import CrossValidationScoreTask, MakePredictionsTask, TrainingDataTask, LoadTestDataTask


class Runner(object):
    def __init__(self, settings_file, targets, pipelines_classifiers, cv_ratio, CLEAN_CACHE=None, DEBUG=False):
        with open(settings_file) as f:
            settings = json.load(f)
        root = os.path.dirname(os.getcwd())
        self.data_dir = os.path.join(root, str(settings['competition-data-dir']))
        if DEBUG:
            self.data_dir = os.path.join(root, str(settings['testing-dir']))
        self.cache_dir = os.path.join(root, str(settings['data-cache-dir']))
        self.submission_dir = os.path.join(root, str(settings['submission-dir']))
        makedirs(self.submission_dir)
        if CLEAN_CACHE:
            CachedDataLoader.clean_cache(self.cache_dir, CLEAN_CACHE)

        self.cached_data_loader = CachedDataLoader(self.cache_dir)
        self.ts = time.get_millis()
        self.targets = targets
        self.pipelines_classifiers = pipelines_classifiers
        self.cv_ratio = cv_ratio
        self.out = []
        self.summaries = []

    def task(self, task_core):
        raise NotImplementedError("Implement this")

    def output(self, task_core):
        raise NotImplementedError("Implement this")

    def summary(self):
        pass

    def concat_train(self, targets_data):
        result = JSDict({'X_train':None, 'y_train':None, 'X_cv': None, 'y_cv':None, 'y_classes': None, 'X': None, 'y': None})
        result.X_train = np.concatenate(tuple([td.X_train for td in targets_data]))
        result.y_train = np.concatenate(tuple([td.y_train for td in targets_data]))
        result.X_cv = np.concatenate(tuple([td.X_cv for td in targets_data]))
        result.y_cv = np.concatenate(tuple([td.y_cv for td in targets_data]))
        result.y_classes = targets_data[0].y_classes
        return result

    def concat_test(self, targets_data):
        result = JSDict({'X': None, 'File': None})
        result.X = np.concatenate(tuple([td.X for td in targets_data]))
        result.File = ','.join([td.File for td in targets_data])
        return result

    def run(self):
        for pipeline, classifier in self.pipelines_classifiers:
            print 'Using pipeline %s with classifier %s' % (pipeline.get_name(), classifier.get_name())
            self.out = []
            task_core = TaskCore(cached_data_loader=self.cached_data_loader, data_dir=self.data_dir,
                                 target=None, pipeline=pipeline, classifier=classifier,
                                 parser=None, cv_ratio=self.cv_ratio,
                                 train=None, test=None)
            for target in self.targets:
                task_core = task_core._replace(target=target, parser=SeizureParser(self.data_dir, target, pipeline))
                self.task(task_core)
            task_core = task_core._replace(target='all')
            self.output(task_core)
            gc.collect()
        self.summary()


class DataPrepRunner(Runner):
    def task(self, task_core):
        data = TrainingDataTask(task_core).run()
        # print data.X_train.shape
        # print data.X_cv.shape

    def output(self, task_core):
        print 'Data preparation done for pipeline %s' % task_core.pipeline.get_name()


class PredictionRunner(Runner):
    def task(self, task_core):
        predictions = MakePredictionsTask(task_core).run()
        result = pd.DataFrame(predictions.file_id.split(","), columns=["File"])
        result["Class"] = predictions.data
        self.out.append(result)

    def output(self, task_core):
        filename = 'submission%d-%s_%s' % (self.ts, task_core.classifier.get_name(), task_core.pipeline.get_name())
        data = pd.concat(self.out)
        Writer(self.submission_dir, filename).write(data)


class CVRunner(Runner):
    def task(self, task_core):
        data = CrossValidationScoreTask(task_core).run()
        self.out.append(data.score)
        print 'score: %.3f' % data.score
        print data.true
        print data.prediction

    def output(self, task_core):
        name = task_core.pipeline.get_name() + '_' + task_core.classifier.get_name()
        summary = get_score_summary(name, self.out)
        self.summaries.append((summary, np.mean(self.out)))
        print summary

    def summary(self):
        print_results(self.summaries)


class EnsembleLearnRunner(Runner):
    #output (prediction, true) tuple for each target, self.out should be a list of tuples
    def task(self, task_core):
        classifier_data = CrossValidationScoreTask(task_core).run()
        self.out.append((classifier_data.prediction, classifier_data.true))

    #output (model_name, [p1, p2, p3]) tuple, self.summaries should be a list of the tuples
    def output(self, task_core):
        name = task_core.pipeline.get_name() + '_' + task_core.classifier.get_name()
        self.summaries.append((name, self.out))

    #compute weights of different models based on the predictions, true values and score metric
    #output weights of different models (for each target)
    def summary(self):
        target_index = 0
        result = []
        for target in self.targets:
            target_data = self.get_target_predictions(target_index)
            result.append(self.compute_model_weights(target_data))
        print result

    def get_target_predictions(self, target_index):
        return [predictions[1][target_index] for predictions in self.summaries]

    def compute_model_weights(self, data):
        predictions = [d[0] for d in data]
        true = data[0][1]

        def f(x, preds):
            pred = np.dot(x, preds)
            return roc_auc_score(true, pred)

        c1 = lambda x : sum(x) - 1
        c2 = lambda x : 1 - sum(x)
        c3 = lambda x : x[0]
        c4 = lambda x : 1 - x[0]

        Initial = np.repeat(1.0/len(predictions), len(predictions))

        opt = fmin_cobyla(f, Initial, cons=[c1, c2, c3, c4], args=predictions)
        pass


class CVAllRunner(Runner):
    def task(self, task_core):
        self.out.append(TrainingDataTask(task_core).run())


    def output(self, task_core):
        name = task_core.pipeline.get_name() + '_' + task_core.classifier.get_name()
        task_core = task_core._replace(train=self.concat_train(self.out))
        summary = CrossValidationScoreTask(task_core).run()
        self.summaries.append((name, summary.score))
        print summary

    def summary(self):
        self.summaries.sort(cmp=lambda x, y: cmp(x[1], y[1]))
        if len(self.summaries) > 1:
            print 'summaries'
            for name, score in self.summaries:
                print name,
                print score


class PredictionAllRunner(Runner):
    def task(self, task_core):
        train = TrainingDataTask(task_core).run()
        test = LoadTestDataTask(task_core).run()
        self.out.append((train, test))

    def output(self, task_core):
        targets_train = [o[0] for o in self.out]
        targets_test = [o[1] for o in self.out]
        task_core = task_core._replace(train=self.concat_train(targets_train), test=self.concat_test(targets_test))
        prediction = MakePredictionsTask(task_core).run()
        result = pd.DataFrame(prediction.file_id.split(","), columns=["File"])
        result["Class"] = prediction.data
        filename = 'submission%d-%s_%s' % (self.ts, task_core.classifier.get_name(), task_core.pipeline.get_name())
        Writer(self.submission_dir, filename).write(result)


