import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from keras.models import Sequential
from common.utils.generator import dict_generator
from common.classifiers import Model


class RandomForest(Model):
    def __init__(self, name, **params):
        super(RandomForest, self).__init__(name)
        self.model = RandomForestClassifier()
        self.model.set_params(**params)

    def fit(self, data, use_all_data):
        if use_all_data:
            self.model.fit(data.X, data.y)
        else:
            self.model.fit(data.X_train, data.y_train)

    def predict_proba(self, x_test):
        predictions = self.model.predict_proba(x_test)
        return predictions[:, 1]

    @staticmethod
    def generate_models(params):
        models = []
        for param in dict_generator(params):
            name = 'RFe%ss%sb%sj%sr%s' % tuple(param.values())
            models.append(RandomForest(name, **param))
        return models


class XGBTree(Model):
    def __init__(self, name, **params):
        super(XGBTree, self).__init__(name)
        self.model = xgb.XGBClassifier()
        self.model.set_params(**params)

    def fit(self, data, use_all_data):
        self.model.fit(data.X_train, data.y_train,
                       eval_set=[(data.X_train, data.y_train), (data.X_cv, data.y_cv)],
                       eval_metric='logloss', early_stopping_rounds=5)
        if use_all_data:
            self.model.set_params(n_estimators=self.model.best_iteration)
            self.model.fit(data.X, data.y)

    def predict_proba(self, x_test):
        predictions = self.model.predict_proba(x_test, ntree_limit=self.model.best_ntree_limit)
        return predictions[:, 1]

    @staticmethod
    def generate_models(params):
        #params are reg_alpha, learning_rate, nthread, n_estimators, subsample, reg_lambda, max_depth, gamma
        models = []
        for param in dict_generator(params):
            name = 'XGBTreed%sl%se%sg%ss%sa%sl%st%s' % tuple(param.values())
            models.append(XGBTree(name, **param))
        return models


class XGBLinear(Model):
    def __init__(self, name, **params):
        super(XGBLinear, self).__init__(name)
        self.model = xgb.XGBRegressor()
        self.model.set_params(**params)

    def fit(self, data, use_all_data):
        self.model.fit(data.X_train, data.y_train,
                       eval_set=[(data.X_train, data.y_train), (data.X_cv, data.y_cv)],
                       eval_metric='logloss', early_stopping_rounds=5)
        if use_all_data:
            self.model.set_params(n_estimators=self.model.best_iteration)
            self.model.fit(data.X, data.y)

    def predict_proba(self, x_test):
        return self.model.predict(x_test, ntree_limit=self.model.best_ntree_limit)

    @staticmethod
    def generate_models(params):
        # params are reg_alpha, learning_rate, nthread, n_estimators, subsample, reg_lambda, max_depth, gamma
        models = []
        for param in dict_generator(params):
            name = 'XGBLineard%sl%se%sg%ss%sa%sl%st%s' % tuple(param.values())
            models.append(XGBLinear(name, **param))
        return models


class NNSeq(Model):
    """
    layer_set should be as [layer, layer ,layer,]
    compile_params and fit_params should be dictionary
    """
    def __init__(self, name, layer_set, compile_params, fit_params):
        super(NNSeq, self).__init__(name)
        self.model = Sequential(layer_set)
        self.model.compile(**compile_params)
        self.fit_params = fit_params

    def batch_generator(self, x, y=None):
        batch_size = self.fit_params['batch_size']
        number_of_batches = np.ceil(x.shape[0] / float(batch_size))
        counter = 0
        sample_index = np.arange(x.shape[0])
        is_train = y is not None
        if is_train:
            np.random.shuffle(sample_index)
        while True:
            batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
            X_batch = x[batch_index, :]
            counter += 1
            if is_train:
                y_batch = y[batch_index]
                yield X_batch, y_batch
            else:
                yield X_batch
            if (counter == number_of_batches):
                if is_train:
                    np.random.shuffle(sample_index)
                counter = 0

    def get_sample_size(self, data_shape):
        batch_size = self.fit_params['batch_size']
        return min(data_shape, batch_size * np.ceil(data_shape / float(batch_size)))

    def fit(self, data, use_all_data):
        if use_all_data:
            self.model.fit_generator(generator=self.batch_generator(x=data.X, y=data.y),
                                     nb_epoch=self.fit_params['nb_epoch'],
                                     samples_per_epoch=self.get_sample_size(data.X.shape[0]),
                                     verbose=self.fit_params['verbose'])
        else:
            self.model.fit_generator(generator=self.batch_generator(x=data.X_train, y=data.y_train),
                                 nb_epoch=self.fit_params['nb_epoch'],
                                 samples_per_epoch=self.get_sample_size(data.X_train.shape[0]),
                                 verbose=self.fit_params['verbose'])

    def predict_proba(self, x_test):
        predictions = self.model.predict_generator(generator=self.batch_generator(x_test),
                                            val_samples=x_test.shape[0])
        return predictions[:, 0]
