import numpy as np


class Classifier(object):
    """
    """
    def __init__(self, model, score_metric):
        self.name = model.get_name()
        self.model = model
        self.score_metric = score_metric

    def get_name(self):
        return self.name

    def get_model(self):
        return self.model

    def fit(self, data, use_all_data):
        self.get_fit_data(data, use_all_data)
        if use_all_data:
            print 'X:', np.shape(data.X)
            print 'y:', np.shape(data.y)
        else:
            print 'X_train:', np.shape(data.X_train)
            print 'y_train:', np.shape(data.y_train)
        self.model.fit(data, use_all_data)

    def get_score(self, data):
        return self.score_metric(self.model, data.X_cv, data.y_cv)

    def get_prediction_prob(self, x_test):
        return self.model.predict_proba(x_test)

    def get_fit_data(self, data, use_all_data):
        if use_all_data:
            data.X = np.concatenate((data.X_train, data.X_cv), axis=0)
            data.y = np.concatenate((data.y_train, data.y_cv), axis=0)


class Model(object):

    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def fit(self, data, use_all_data):
        raise NotImplementedError("Implement this")

    def predict_proba(self, x_test):
        raise NotImplementedError("Implement this")

