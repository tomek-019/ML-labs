import numpy as np
import matplotlib.pyplot as plt


def main():
    data = np.loadtxt('iris.csv', delimiter=',', dtype=object)
    
    column_names = data[0]
    data = data[1:]
    
    features = data[:,:-1]
    labels = data[:,-1]

    labels[labels=='"Setosa"'] = 0
    labels[labels=='"Versicolor"'] = 1
    labels[labels=='"Virginica"'] = 2

    features = features.astype(float)

    fig, axs = plt.subplots(4,4,figsize=(10,10))

    for i in range(4):
        for j in range(4):
            ax = axs[i, j]
            ax.scatter(features[:,i], features[:,j], c=labels, cmap='brg', s=3)
            ax.set_xlabel(str(column_names[i]).strip('"'))
            ax.set_ylabel(str(column_names[j]).strip('"'))
            ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig('iris2.png')

if __name__ == '__main__':
    main()