import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


X, y= make_classification(n_samples=700, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, flip_y=0.08)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = GaussianNB()
clf.fit(X_train, y_train)
proba_t = clf.predict_proba(X_test)
t = np.argmax(proba_t, axis=1)
#print(proba_t)
#print(t)

accuracy = accuracy_score(y_test, t)
print(f'accuracy')

fig, ax = plt.subplots(1, 2, figsize=(16,8))

ax[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test)
ax[1].scatter(X_test[:, 0], X_test[:, 1], c=t)

accuracy_text = f'Dokładność klasyfikacji: {accuracy:.2f}%'

plt.suptitle(accuracy_text)

plt.tight_layout()
ax[0].grid()
ax[1].grid()



plt.savefig('out')

