from __future__ import print_function
from builtins import *
import warnings
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
import ray
import models
from scipy import stats
import copy


def make_pipeline(model):
    steps = list()
    steps.append(('standardize', StandardScaler()))
    steps.append(('normalize', MinMaxScaler()))
    steps.append(('model', model))
    pipeline = Pipeline(steps=steps)
    return pipeline


def evaluate_model(X, Y, model, folds, metric):
    pipeline = make_pipeline(model)
    scores = cross_val_score(pipeline, X, Y, scoring=metric, cv=folds,
                             n_jobs=-1)
    return scores

@ray.remote
def robust_evaluate_modelr(X, Y, model, folds, metric, name):
    scores = None
    print("Evaluating %s" % name)
    #try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        scores = evaluate_model(X, Y, model, folds, metric)
    #except:
    #    scores = None
    return scores

def robust_evaluate_model(X, Y, model, folds, metric, name):
    scores = None
    print("Evaluating %s" % name)
    #try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        scores = evaluate_model(X, Y, model, folds, metric)
    #except:
    #    scores = None
    return scores

def cluster_accuracy(labels_true, labels_pred):
    # find all unique cluster indices
    #print(np.unique(labels_pred), np.unique(labels_true))
    labels_true = labels_true.values
    likely_labels = copy.copy(labels_true)
    for label in np.unique(labels_pred):
        idx = np.argwhere(labels_pred == label).flatten()
        #print(idx)
        #print(labels_true[idx])
        alllabs = list(labels_true[idx])
        likely_label = max(set(alllabs), key=alllabs.count)
        for i in idx:
            likely_labels[i] = likely_label
    #print(likely_labels, labels_true)
    ca = metrics.accuracy_score(labels_true, likely_labels)
    return ca

class spot(object):
    """Docsting."""
    def __init__(self, local=False, data_type='tabular', problem_type='classification'):
        """Docstring."""
        # initialize the parallelism
        self.local = local
        if not local:
            ray.init(num_cpus=4)
        if problem_type == 'classification':
            self.models = models.classifiers
            self.metric = metrics.make_scorer(metrics.accuracy_score)
        elif problem_type == 'clustering':
            self.models = models.clusterers
            self.metric = metrics.make_scorer(cluster_accuracy)#metrics.make_scorer(metrics.adjusted_rand_score)

    def load_data(self, x, y):
        self.x = x
        self.y = y
        return self

    def summarize(self):
        print(self.x.describe())
        # calculate MI between classes
        #print(self.y.describe())
        return self

    def check(self, folds=10, log=True):
        self.results = {}
        if self.local:
            _scores = []
            for name, model in self.models.items():
                scores = robust_evaluate_model(self.x, self.y, model,
                                               folds, self.metric, name)
                if scores is not None:
                    self.results[name] = scores
                    mean_score, std_score = np.mean(scores), np.std(scores)
                    if log:
                        print('Mean: %.3f (+/- %.3f)' % (mean_score, std_score))
        else:
            _scores = []
            for name, model in self.models.items():
                _scores.append(robust_evaluate_modelr.remote(self.x, self.y, model,
                                                            folds, metric, name))
            print(_scores)
            __scores = ray.get(_scores)
            for _score, (name, model) in zip(__scores, self.models.items()):
                scores = _score
                if scores is not None:
                    self.results[name] = scores
                    mean_score, std_score = np.mean(scores), np.std(scores)
                    print('Mean: %.3f (+/- %.3f)' % (mean_score, std_score))
        return self

    def summarize_check(self, maximize=True, top_n=10):
        if len(self.results) == 0:
            print("No Results!")
            return
        n = min(top_n, len(self.results))
        mean_scores = [np.mean(v) for k, v in self.results.items()]
        names = [k for k, v in self.results.items()]
        idx = np.argsort(mean_scores).flatten()
        if maximize:
            idx = list(reversed(idx))
        mean_scores = [mean_scores[i] for i in idx]#mean_scores[idx]
        names = [names[i] for i in idx]
        names = names[:n]
        scores = mean_scores[:n]
        print()
        for i in range(n):
            name = names[i]
            mean_score, std_score = \
                np.mean(self.results[name]), np.std(self.results[name])
            print('Rank=%d, Name=%s, Score=%.3f (+/- %.3f)' % (i+1, name, mean_score, std_score))
        return self
