import numpy as np
from sklearn.metrics import roc_auc_score


# helper methods for printing scores
def get_score_summary(name, scores):
    summary = '%.3f-%.3f (mean=%.5f std=%.5f)' % (min(scores), max(scores), np.mean(scores), np.std(scores))
    score_list = ['%.3f' % score for score in scores]
    return '%s %s [%s]' % (name, summary, ','.join(score_list))


def print_results(summaries, reverse=False):
    summaries.sort(cmp=lambda x,y: cmp(x[1], y[1]), reverse=reverse)
    if len(summaries) > 1:
        print 'summaries'
        for s, mean in summaries:
            print s


def auc_score(model, X, y):
    prediction = model.predict_proba(X)
    return roc_auc_score(y, prediction)
