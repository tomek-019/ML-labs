from sklearn.datasets import load_digits
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

digits = load_digits()

x = digits.data
y = digits.target

fx=x[0]
fy=y[0]
print(f"{fx}\n")
fx=fx.reshape(8, 8)
print(fx, f"Etykieta: {fy}\n", sep="\n\n")

n_splits = 2
n_repeats = 5

classifiers = [GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier()]
clf_names = ["GNB", "KNN", "DT"]

for idx, clf in enumerate(classifiers):
    name = clf_names[idx]
    scores = []

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)

        for train_idx, test_idx in skf.split(x, y):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)

            score = accuracy_score(y_test, y_pred)
            scores.append(score)

    mean = np.mean(scores)
    std = np.std(scores)
    print(f"{name} {mean:.3f} ({std:.3f})")
