import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import ttest_rel


data = np.loadtxt('task1.csv', delimiter=',', dtype=object)

x = data[:, :-1].astype(float)
y = data[:, -1]

classifiers = [GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier()]
clf_names = ["GNB", "KNN", "DT"]

alpha = 0.05
n_splits = 2
n_repeats = 5
scores = np.zeros((len(classifiers), n_splits * n_repeats))
t_stat = np.zeros((len(classifiers), len(classifiers)))
p_val = np.zeros((len(classifiers), len(classifiers)))

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

for idx1, name1 in enumerate(clf_names):
    for idx2, name2 in enumerate(clf_names):
        result = ttest_rel(scores[idx1, :], scores[idx2, :])
        t_stat[idx1, idx2] = result.statistic
        p_val[idx1, idx2] = result.pvalue

print(f"t-statistic matrix:\n{t_stat}\n")
print(f"p-value matrix:\n{p_val}\n")

print(f"better matrix:\n{t_stat > 0}\n")
print(f"significant matrix:\n{p_val < alpha}\n")

for idx1, name1 in enumerate(clf_names):
    for idx2, name2 in enumerate(clf_names):
        if idx2 > idx1:
            if t_stat[idx1, idx2] > 0 and p_val[idx1, idx2] < alpha:
                print(f"{name1} ze średnią {np.mean(scores[idx1, :]):.3f} jest statystycznie znacząco lepszy niż {name2} ze średnią {np.mean(scores[idx2, :]):.3f}\n")
            elif t_stat[idx1, idx2] < 0 and p_val[idx1, idx2] < alpha:
                print(f"{name2} ze średnią {np.mean(scores[idx2, :]):.3f} jest statystycznie znacząco lepszy niż {name1} ze średnią {np.mean(scores[idx1, :]):.3f}\n")
            else:
                print(f"Brak różnicy statystycznej między {name1} ze średnią {np.mean(scores[idx1, :]):.3f} a {name2} ze średnią {np.mean(scores[idx2, :]):.3f}\n")