import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


X, y= make_classification(n_samples=700, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, flip_y=0.08)

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
rskf.get_n_splits()
list = []

for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    t = clf.predict(X_test)
    accuracy = accuracy_score(y_test, t)
    list.append(accuracy)

list_np = np.array(list)

print(f'Wszystkie wyniki:\n{list}')
print(f'Oczekiwana wartość: {np.mean(list_np):.3f}')
print(f'Odchylenie standardowe: {np.std(list_np):.3f}')
