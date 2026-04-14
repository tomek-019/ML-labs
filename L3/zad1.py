import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

X, y= make_classification(n_samples=700, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, flip_y=0.08)

data = np.column_stack((X, y))

np.savetxt('data.csv', data)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.tight_layout()
plt.grid()

plt.savefig('out')