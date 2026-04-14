import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

class MajorityClassifier(ClassifierMixin, BaseEstimator):

    def fit(self, x, y):
        unique, counts = np.unique(y, return_counts=True)
        self.majority_class_ = unique[np.argmax(counts)]
        return self

    def predict(self, x):
        return np.full(len(x), self.majority_class_)


def main():
    
    x, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, weights=[0.8, 0.2], random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=430)

    clf = MajorityClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
 
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Balanced Accuracy Score: {balanced_accuracy_score(y_test, y_pred):.3f}")

if __name__ == '__main__':
    main()
