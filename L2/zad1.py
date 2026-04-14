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

    fig, ax = plt.subplots(1,1,figsize=(7,7))
    ax.scatter(features[:,0], features[:,1], c=labels, cmap='brg')
    ax.set_xlabel(str(column_names[0]).strip('"'))
    ax.set_ylabel(str(column_names[1]).strip('"'))
    fig.tight_layout()
    fig.savefig('iris1.png')

if __name__ == '__main__':
    main()