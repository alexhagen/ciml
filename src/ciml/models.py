from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np

# Data is "image-like"
    # Task is "classification"

    # Task is "regression"

# Data is "series-like"
    # Task is "classification"

    # Task is "regression"

# Data is "3d-like"
    # Task is "classification"

    # Task is "regression"

# Data is "tree-like"
    # Task is "classification"

    # Task is "regression"

# Data is "unspecified"
    # Task is "classification"

    # Task is "regression"


class model_def(dict):
    """Docstring."""
    def __init__(self, key, constructor):
        """Docstring."""
        pass

# Taken from Jason Brownlee -- put his license here
classifiers = {}
classifiers['logistic'] = LogisticRegression()
alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for a in alpha:
    classifiers['ridge'+str(a)] = RidgeClassifier(alpha=a)
classifiers['sgd'] = SGDClassifier(max_iter=1000, tol=1e-3)
classifiers['pa'] = PassiveAggressiveClassifier(max_iter=1000, tol=1e-3)
# non-linear models
n_neighbors = range(1, 21)
for k in n_neighbors:
    classifiers['knn'+str(k)] = KNeighborsClassifier(n_neighbors=k)
classifiers['cart'] = DecisionTreeClassifier()
classifiers['extra'] = ExtraTreeClassifier()
classifiers['svml'] = SVC(kernel='linear')
classifiers['svmp'] = SVC(kernel='poly')
c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for c in c_values:
    classifiers['svmr'+str(c)] = SVC(C=c)
classifiers['bayes'] = GaussianNB()
# ensemble models
n_trees = 100
classifiers['ada'] = AdaBoostClassifier(n_estimators=n_trees)
classifiers['bag'] = BaggingClassifier(n_estimators=n_trees)
classifiers['rf'] = RandomForestClassifier(n_estimators=n_trees)
classifiers['et'] = ExtraTreesClassifier(n_estimators=n_trees)
classifiers['gbm'] = GradientBoostingClassifier(n_estimators=n_trees)
for hls in [(100,), (10, 10), (8, 8, 8, 8, 8)]:
    classifiers['mlp' + '/'.join([str(i) for i in hls])] = MLPClassifier(hidden_layer_sizes=hls)


DBSCAN.predict = DBSCAN.fit_predict
AgglomerativeClustering.predict = AgglomerativeClustering.fit_predict
clusterers = {}
for eps in np.linspace(0.1, 250.0, 10):
    clusterers[('dbscan%.2f' % eps)] = DBSCAN(eps=eps)
for k in range(5, 50, 5):
    clusterers['kmeans'+str(k)] = KMeans(n_clusters=k)
for k in range(5, 50, 5):
    clusterers['gm' + str(k)] = GaussianMixture(n_components=k)
for k in range(5, 50, 5):
    clusterers['ac' + str(k)] = AgglomerativeClustering(n_clusters=2)
clusterers['meanshift'] = MeanShift()
