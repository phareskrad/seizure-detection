import json
import os

from common.classifiers import Classifier
from common.data import CachedDataLoader, makedirs
from common.pipeline import Pipeline
from common.utils import time
from seizure.io import Parser
from seizure.models import RandomForest
from seizure.scores import auc_score
from seizure.tasks import TaskCore
from seizure.transforms import Stats

with open('settings.json') as f:
    settings = json.load(f)
DEBUG = False
CLEAN_CACHE = None
root = os.path.dirname(os.getcwd())
data_dir = os.path.join(root, str(settings['competition-data-dir']))
if DEBUG:
    data_dir = os.path.join(root, str(settings['testing-dir']))
cache_dir = os.path.join(root, str(settings['data-cache-dir']))
submission_dir = os.path.join(root, str(settings['submission-dir']))
makedirs(submission_dir)
if CLEAN_CACHE:
    CachedDataLoader.clean_cache(cache_dir, CLEAN_CACHE)

cached_data_loader = CachedDataLoader(cache_dir)
ts = time.get_millis()
targets = ['1','2','3']
pipeline = Pipeline(transform=[Stats()])
params = {'n_estimators': [400], 'min_samples_split': [1], 'bootstrap': [True], 'n_jobs': [5],
           'random_state': [0]}
classifiers = [Classifier(model, auc_score) for model in RandomForest.generate_models(params)]
classifier = classifiers[0]
cv_ratio = .5
out = []
summaries = []

task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                 target='1', pipeline=pipeline, classifier=classifier,
                                 parser=Parser(data_dir, '1', pipeline), cv_ratio=cv_ratio,
                                 train=None, test=None)

