from sklearn.datasets import load_digits
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
import warnings

warnings.filterwarnings('ignore')

def fit_predict_save(clf, x_train, y_train, x_test, y_test):
    scores = []
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
    return scores

digits = load_digits()

x = digits.data
y = digits.target

n_splits = 2
n_repeats = 5
head = True
show_pca = True
show_kbest = True

classifiers = [GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier(random_state=42)]
clf_names = ["GNB", "KNN", "DT"]
results = {}

for idx, clf in enumerate(classifiers):
    name = clf_names[idx]
    scores_base = []
    scores_norm = []
    scores_pca = []
    scores_kbest = []

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)

        for train_idx, test_idx in skf.split(x, y):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scores_base += fit_predict_save(clf, x_train, y_train, x_test, y_test)

            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train_scaled = scaler.transform(x_train)
            x_test_scaled = scaler.transform(x_test)

            scores_norm += fit_predict_save(clf, x_train_scaled, y_train, x_test_scaled, y_test)

            pca = PCA(n_components=0.8)
            pca.fit(x_train)
            x_train_pca = pca.transform(x_train)
            x_test_pca = pca.transform(x_test)
            
            if idx == 0 and repeat == 0 and show_pca:
                print(f"Pierwszy obiekt po PCA:\n{x_train_pca[0]}\n")
                show_pca = False

            scores_pca += fit_predict_save(clf, x_train_pca, y_train, x_test_pca, y_test)

            kbest = SelectKBest(k=int(np.sqrt(x.shape[1])))
            kbest.fit(x_train, y_train)
            x_train_kbest = kbest.transform(x_train)
            x_test_kbest = kbest.transform(x_test)

            if idx == 0 and repeat == 0 and show_kbest:
                print(f"Pierwszy obiekt po KBest:\n{x_train_kbest[0]}\n\nEtykieta: {y_train[0]}\n")
                show_kbest = False

            scores_kbest += fit_predict_save(clf, x_train_kbest, y_train, x_test_kbest, y_test)

    results[name] = {'base': scores_base, 'norm': scores_norm, 'pca':  scores_pca, 'kbest': scores_kbest}

print("PCA")
for name in clf_names:
    print(f"{name} {np.mean(results[name]['pca']):.3f} ({np.std(results[name]['pca']):.3f})")

print("\nSelect K-best")
for name in clf_names:
    print(f"{name} {np.mean(results[name]['kbest']):.3f} ({np.std(results[name]['kbest']):.3f})")

print(f"\n{'Clf':<6} {'Base':<6} {'Norm':<6} {'PCA':<6} {'KBest':<6}")
for name in clf_names:
    print(f"{name:<5}  {np.mean(results[name]['base']):.3f}  {np.mean(results[name]['norm']):.3f}  {np.mean(results[name]['pca']):.3f}  {np.mean(results[name]['kbest']):.3f}")
