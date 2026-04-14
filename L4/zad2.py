import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier

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
    
    x, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, weights=[0.8, 0.2], random_state=42)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=430)

    clf = KNN(k=5)
    clf2 = KNeighborsClassifier(n_neighbors=5, algorithm='brute')

    clf.fit(x_train, y_train)
    clf2.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_pred2 = clf2.predict(x_test) 

    print(f"Custom KNN  | Balanced Accuracy Score: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(f"Sklearn KNN | Balanced Accuracy Score: {balanced_accuracy_score(y_test, y_pred2):.3f}")

if __name__ == '__main__':
    main()