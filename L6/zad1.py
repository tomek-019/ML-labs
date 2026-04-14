import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import shapiro


data = np.loadtxt('task1.csv', delimiter=',', dtype=object)

x = data[:, :-1].astype(float)
y = data[:, -1]

classifiers = [GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier()]
clf_names = ["GNB", "KNN", "DT"]

alpha = 0.05
n_splits = 2
n_repeats = 5
scores = np.zeros((len(classifiers), n_splits * n_repeats))

for idx_clf, clf in enumerate(classifiers):
    name = clf_names[idx_clf]

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

    for idx_fold, (train_idx, test_idx) in enumerate(rskf.split(x, y)):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        score = balanced_accuracy_score(y_test, y_pred)
        scores[idx_clf, idx_fold] = score

for idx_clf, name in enumerate(clf_names):
    clf_scores = scores[idx_clf, :]
    stat, p = shapiro(clf_scores)
    print(f"Kladyfikator: {name} | statistic: {stat:.3f} | p-value: {p:.3f}")
    if p > alpha:
        print("Rozkład normalny: True\n")
    else:
        print("Rozkład normalny: False\n")

    



