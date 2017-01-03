from common.pipeline import Pipeline
from seizure.transforms import FFT, Slice, Magnitude, Log10, FFTWithTimeFreqCorrelation, MFCC, Resample, Stats, \
    DaubWaveletStats, TimeCorrelation, FreqCorrelation, TimeFreqCorrelation
from seizure.runs import PredictionAllRunner, CVAllRunner, DataPrepRunner
from seizure.models import RandomForest, XGBTree, XGBLinear, NNSeq
from common.classifiers import Classifier
from common.utils.generator import tuple_generator
from seizure.scores import auc_score
from keras.layers.advanced_activations import PReLU, ELU
from keras.layers import Dense, Dropout, Activation

targets = [
    '1',
    '2',
    '3'
]
pipelines = [
    # NOTE(mike): you can enable multiple pipelines to run them all and compare results
    Pipeline(transform=[FFT(), Slice(1, 48), Magnitude(), Log10()]),
    Pipeline(transform=[FFT(), Slice(1, 64), Magnitude(), Log10()]),
    Pipeline(transform=[FFT(), Slice(1, 96), Magnitude(), Log10()]),
    # Pipeline(transform=[FFT(), Slice(1, 128), Magnitude(), Log10()]),
    # Pipeline(transform=[FFT(), Slice(1, 160), Magnitude(), Log10()]),
    Pipeline(transform=[Stats()]), #48
    Pipeline(transform=[DaubWaveletStats(4)]), #576
    Pipeline(transform=[Resample(400), DaubWaveletStats(2)]), #320
    Pipeline(transform=[Resample(400), MFCC()]), #208
    Pipeline(transform=[FFTWithTimeFreqCorrelation(1, 48, 400, 'us')]), #1024
    Pipeline(transform=[FFTWithTimeFreqCorrelation(1, 48, 400, 'usf')]), #1024
    # winning submission # higher score than winning submission
    Pipeline(transform=[FFTWithTimeFreqCorrelation(1, 48, 400, 'none')]), #1024
    Pipeline(transform=[TimeCorrelation(400, 'usf', with_corr=True, with_eigen=True)]), #136
    Pipeline(transform=[TimeCorrelation(400, 'us', with_corr=True, with_eigen=True)]), #136
    Pipeline(transform=[TimeCorrelation(400, 'us', with_corr=True, with_eigen=False)]), #120
    Pipeline(transform=[TimeCorrelation(400, 'us', with_corr=False, with_eigen=True)]), #16
    Pipeline(transform=[TimeCorrelation(400, 'none', with_corr=True, with_eigen=True)]), #136
    Pipeline(transform=[FreqCorrelation(1, 48, 'usf', with_corr=True, with_eigen=True)]), #136
    Pipeline(transform=[FreqCorrelation(1, 48, 'us', with_corr=True, with_eigen=True)]), #136
    Pipeline(transform=[FreqCorrelation(1, 48, 'us', with_corr=True, with_eigen=False)]), #120
    Pipeline(transform=[FreqCorrelation(1, 48, 'us', with_corr=False, with_eigen=True)]), #16
    Pipeline(transform=[FreqCorrelation(1, 48, 'none', with_corr=True, with_eigen=True)]), #136
    Pipeline(transform=[TimeFreqCorrelation(1, 48, 400, 'us')]), #272
    Pipeline(transform=[TimeFreqCorrelation(1, 48, 400, 'usf')]), #272
    Pipeline(transform=[TimeFreqCorrelation(1, 48, 400, 'none')]), #272
]
# params = {'n_estimators': [100, 200, 300, 400], 'min_samples_split': [1, 2], 'bootstrap': [True, False], 'n_jobs': [5],
#           'random_state': [0]}
params = {'n_estimators': [300], 'min_samples_split': [1], 'bootstrap': [True], 'n_jobs': [5],
          'random_state': [0]}

classifiers = [Classifier(model, auc_score) for model in RandomForest.generate_models(params)]
# time-correlation-r400-us-noeig_RFe300s1bTruej5r0 0.838608262798
params = {'max_depth': [4, 3, 5, 6],
          'learning_rate': [0.3, 0.2, 0.1],
          'n_estimators': [200],
          'gamma': [0.2, 0.1, 0.3, 0.4],
          'subsample': [1, 0.9, 0.7],
          'reg_alpha': [0],
          'reg_lambda': [1],
          'nthread':[4]}

# params = {'max_depth': [6],
#           'learning_rate': [0.3],
#           'n_estimators': [200],
#           'gamma': [0.2],
#           'subsample': [1],
#           'reg_alpha': [0],
#           'reg_lambda': [1],
#           'nthread':[4]}
classifiers = [Classifier(model, auc_score) for model in XGBTree.generate_models(params)]
# time-correlation-r400-us-noeig_XGBTreed0l0.2e4g200s0.7a1l6t0.1 0.835530536671
params = {'max_depth': [6], 'learning_rate': [0.1, 0.2, 0.3], 'n_estimators': [200, 300, 400, 500],
         'gamma': [0.1, 0.2, 0.3], 'subsample': [1], 'reg_alpha': [0.05, 0.25, 0.5, 1],
         'reg_lambda': [1], 'nthread':[4]}
classifiers = [Classifier(model, auc_score) for model in XGBLinear.generate_models(params)]
# time-correlation-r400-us-noeig_XGBLineard0.5l0.2e4g200s1a1l6t0.2 0.821348374679

# classifiers = [
#     #Classifier(XGBTree("XGBTreetest", max_depth=3, learning_rate=0.05, n_estimators=100), auc_score),
#     #Classifier(XGBLinear("XGBLineartest", max_depth=3, learning_rate=0.05, n_estimators=300), auc_score)
#     #Classifier(XGBLinear("NNtest", [Dense(),], {'loss':'binary_crossentropy', 'optimizer':'adam'}, {'nb_epoch': 16, 'batch_size':100, 'verbose': 2}), auc_score)
# ]
layers1 = [Dense(1000, input_dim=136, init='normal'), PReLU(), Dropout(0.4),
           Dense(100), PReLU(), Dropout(0.2),
           Dense(1, init='normal', activation='sigmoid'),]
layers2 = [Dense(1000, input_dim=136, init='normal'), PReLU(), Dropout(0.4),
           Dense(100), PReLU(), Dropout(0.2),
           Dense(10), PReLU(), Dropout(0.1),
           Dense(1, init='normal', activation='sigmoid'),]
layers3 = [Dense(1500, input_dim=136, init='normal'), Activation('tanh'), Dropout(0.4),
           Dense(50), Activation('tanh'), Dropout(0.2),
           Dense(1, init='normal', activation='sigmoid'),]
layers4 = [Dense(1000, input_dim=136, init='normal'), Activation('softplus'), Dropout(0.4),
           Dense(100), Activation('softplus'), Dropout(0.2),
           Dense(1, init='normal', activation='sigmoid'),]
layers5 = [Dense(1000, input_dim=136, init='normal'), Activation('sigmoid'), Dropout(0.4),
           Dense(100), Activation('sigmoid'), Dropout(0.2),
           Dense(1, init='normal', activation='sigmoid'),]
compile_params = {'loss': 'binary_crossentropy', 'optimizer':'adam'}
fit_params = {'nb_epoch': 20, 'batch_size': 100, 'verbose': 2}
NN1 = NNSeq("NNd1000prelud.4d100prelud.2", layers1, compile_params, fit_params)
NN2 = NNSeq("NNd1000prelud.4d100prelud.2d10prelud.1", layers2, compile_params, fit_params)
NN3 = NNSeq("NNd1500tanhd.4d50tanhd.2", layers3, compile_params, fit_params)
NN4 = NNSeq("NNd1000softplusd.4d100softplusd.2", layers4, compile_params, fit_params)
NN5 = NNSeq("NNd1000sigmoidd.4d100sigmoidd.2", layers5, compile_params, fit_params)

classifiers = [Classifier(NN1, auc_score),
               Classifier(NN2, auc_score),
               Classifier(NN3, auc_score),
               Classifier(NN4, auc_score),
               Classifier(NN5, auc_score)]
# time-correlation-r400-us_NNd1500tanhd.4d50tanhd.2 0.838237528702
# time-freq-correlation-1-48-r400-none_NNd1000prelud.4d100prelud.2 0.89191
# time-correlation-r400-us-noeig_NNd1000prelud.4d100prelud.2 0.88288


def run_seizure_detection(build_target, targets=targets, pipelines_classifiers=tuple_generator(pipelines, classifiers)):
    """
    The main entry point for running seizure-detection cross-validation and predictions.
    Directories from settings file are configured, classifiers are chosen, pipelines are
    chosen, and the chosen build_target ('cv', 'predict', 'train_model') is run across
    all combinations of (targets, pipelines, classifiers)
    """
    cv_ratio = 0.5

    if build_target == 'cv':
        CVAllRunner('SETTINGS.json', targets, pipelines_classifiers, cv_ratio).run()
    elif build_target == 'prediction':
        PredictionAllRunner('SETTINGS.json', targets, pipelines_classifiers, cv_ratio).run()
    elif build_target == 'data_prep':
        DataPrepRunner('SETTINGS.json', targets, pipelines_classifiers, cv_ratio).run()
    else:
        raise Exception("unknown build target %s" % build_target)


run_seizure_detection('data_prep')