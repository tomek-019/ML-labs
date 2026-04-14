import numpy as np
from sklearn.datasets import make_classification, make_moons, make_circles, make_gaussian_quantiles
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.naive_bayes import GaussianNB

class KNN(ClassifierMixin, BaseEstimator):
    def __init__(self, k=5, random_state=None):
        self.k_ = k
        self.random_state_ = random_state
        self.random_ = np.random.RandomState(seed=self.random_state_)

    def fit(self, x, y):
        self.x_, self.y_ = x, y
        self.labels_ = np.unique(self.y_)
        return self

    def predict(self, x):
        distance_matrix = np.argsort(cdist(x, self.x_), axis=1)[:, :self.k_]
        #print(distance_matrix)

        k_labels = self.y_[distance_matrix]
        #print(k_labels, k_labels.shape)

        preds, _= mode(k_labels, axis=1)
        #print(preds)
        return preds


def main():

    datasets = [
        make_classification(weights=[0.8, 0.2]),
        make_moons(),
        make_circles(),
        make_gaussian_quantiles(n_classes=4)
    ]
    dataset_names = ['make_classification', 'make_moons', 'make_circles', 'make_gaussian_quantiles']

    classifiers = [KNN(k=5), GaussianNB()]
    clf_names = ['KNN', 'GNB']

    n_datasets = len(datasets)
    n_splits = 2
    n_repeats = 5
    n_folds = n_splits * n_repeats
    n_classifiers = len(classifiers)

    results = np.zeros((n_datasets, n_folds, n_classifiers))

    for d_idx, (x, y) in enumerate(datasets):
        fold_idx = 0
        for repeat in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
            for train_idx, test_idx in skf.split(x, y):
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                for c_idx, clf in enumerate(classifiers):
                    clf.fit(x_train, y_train)
                    y_pred = clf.predict(x_test)
                    results[d_idx, fold_idx, c_idx] = balanced_accuracy_score(y_test, y_pred)

                fold_idx += 1

    for d_idx, name in enumerate(dataset_names):
        print(f"\n{name}:")
        for c_idx, clf_name in enumerate(clf_names):
            scores = results[d_idx, :, c_idx]
            print(f"  {clf_name}: {np.mean(scores):.3f} ({np.std(scores):.3f})")


if __name__ == '__main__':
    main()