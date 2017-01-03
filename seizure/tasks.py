import random
import numpy as np
import common.utils.time as time
from common.task import Task


class LoadIctalDataTask(Task):
    """
    Load the ictal mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X, 'Y': y}
    """
    def filename(self):
        return 'data_ictal_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return self.task_core.parser.parse('ictal')


class LoadInterictalDataTask(Task):
    """
    Load the interictal mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X, 'Y': y}
    """
    def filename(self):
        return 'data_interictal_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return self.task_core.parser.parse('interictal')


class LoadTestDataTask(Task):
    """
    Load the test mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X}
    """
    def filename(self):
        return 'data_test_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return self.task_core.parser.parse('test')


class TrainingDataTask(Task):
    """
    Creating a training set and cross-validation set from the transformed ictal and interictal data.
    """
    def filename(self):
        return None  # not cached, should be fast enough to not need caching

    def load_data(self):
        ictal_data = LoadIctalDataTask(self.task_core).run()
        interictal_data = LoadInterictalDataTask(self.task_core).run()
        return prepare_training_data(ictal_data, interictal_data, self.task_core.cv_ratio)


class CrossValidationScoreTask(Task):
    """
    Run a classifier over a training set, and give a cross-validation score.
    """
    def filename(self):
        return 'score_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier.get_name())

    def load_data(self):
        if self.task_core.train is None:
            self.task_core = self.task_core._replace(train=TrainingDataTask(self.task_core).run())
        classifier_data = train_classifier(self.task_core.classifier, self.task_core.train)
        return classifier_data


class TrainClassifierTask(Task):
    """
    Run a classifier over the complete data set (training data + cross-validation data combined)
    and save the trained models.
    """
    def filename(self):
        return None

    def load_data(self):
        if self.task_core.train is None:
            self.task_core = self.task_core._replace(train=TrainingDataTask(self.task_core).run())
        return train_classifier(self.task_core.classifier, self.task_core.train, use_all_data=True)


class MakePredictionsTask(Task):
    """
    Make predictions on the test data.
    """
    def filename(self):
        return 'predictions_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier.get_name())

    def load_data(self):
        classifier_data = TrainClassifierTask(self.task_core).run()
        if self.task_core.test is None:
            self.task_core = self.task_core._replace(test=LoadTestDataTask(self.task_core).run())
        self.task_core.test.X = flatten(self.task_core.test.X)
        print 'X:', self.task_core.test.X.shape

        return make_predictions(self.task_core.test, classifier_data)


# flatten data down to 2 dimensions for putting through a classifier
def flatten(data):
    if data.ndim > 2:
        return data.reshape((data.shape[0], np.product(data.shape[1:])))
    else:
        return data


def concat(a, b):
    return np.concatenate((a, b), axis=0)


# split up ictal and interictal data into training set and cross-validation set
def prepare_training_data(ictal_data, interictal_data, cv_ratio):
    print 'Preparing training data ...',
    ictal_X, ictal_y = flatten(ictal_data.X), ictal_data.y
    interictal_X, interictal_y = flatten(interictal_data.X), interictal_data.y

    # split up data into training set and cross-validation set for both seizure and early sets
    ictal_X_train, ictal_y_train, ictal_X_cv, ictal_y_cv = split_train_random(ictal_X, ictal_y, ictal_data.sequence_id, cv_ratio)
    interictal_X_train, interictal_y_train, interictal_X_cv, interictal_y_cv = split_train_random(interictal_X, interictal_y, interictal_data.sequence_id, cv_ratio)

    X_train = concat(ictal_X_train, interictal_X_train)
    y_train = concat(ictal_y_train, interictal_y_train)
    X_cv = concat(ictal_X_cv, interictal_X_cv)
    y_cv = concat(ictal_y_cv, interictal_y_cv)

    y_classes = np.unique(concat(y_train, y_cv))

    start = time.get_seconds()
    elapsedSecs = time.get_seconds() - start
    print "%ds" % int(elapsedSecs)

    print 'X_train:', np.shape(X_train)
    print 'y_train:', np.shape(y_train)
    print 'X_cv:', np.shape(X_cv)
    print 'y_cv:', np.shape(y_cv)
    print 'y_classes:', y_classes

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_cv': X_cv,
        'y_cv': y_cv,
        'y_classes': y_classes,
        'X': None,
        'y': None
    }


# split interictal segments at random for training and cross-validation
def split_train_random(X, y, sequence_id, cv_ratio):
    random.seed(2)
    unique_seq = set(sequence_id)
    cv_seq = random.sample(unique_seq, int(len(unique_seq) * cv_ratio))
    train_seq = [s for s in unique_seq if s not in cv_seq]
    cv_seq.sort()
    train_seq.sort()

    def get_data(data, data_index, index):
        ind = []
        for i in range(len(data_index)):
            if data_index[i] in index:
                ind.append(i)
        if data.ndim > 1:
            return data[ind, :]
        else:
            return data[ind]

    X_train = get_data(X, sequence_id, train_seq)
    X_cv = get_data(X, sequence_id, cv_seq)
    y_train = get_data(y, sequence_id, train_seq)
    y_cv = get_data(y, sequence_id, cv_seq)

    return X_train, y_train, X_cv, y_cv


def train_classifier(classifier, data, use_all_data=False):
    print "Fitting model ..."
    start = time.get_seconds()
    classifier.fit(data, use_all_data)
    elapsedSecs = time.get_seconds() - start
    print "t=%ds" % int(elapsedSecs)
    if not use_all_data:
        return {
            'true': data.y_cv,
            'prediction': classifier.get_prediction_prob(data.X_cv),
            'score': classifier.get_score(data)
        }
    else:
        return {
            'model': classifier.get_model()
        }


# use the classifier and make predictions on the test data
def make_predictions(test_data, classifier_data):
    model = classifier_data.model
    predictions_proba = model.predict_proba(test_data.X)

    return {
       'data': predictions_proba,
        'file_id': test_data.File
    }
