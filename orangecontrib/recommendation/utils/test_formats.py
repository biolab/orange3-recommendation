import Orange
from orangecontrib.recommendation.utils import format_data

from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict

def tab_format():
    # Load data
    filename = '/Users/salvacarrion/Desktop/trainingSet.csv'
    data = Orange.data.Table(filename)

    A = csr_matrix(data.X)

    indices = A.nonzero()
    for i in range(0, len(indices[0])):
        print('%d\t%d\t%d' % (indices[0][i], indices[1][i], 1))


def print_dense_matrix():
    # Load data
    filename = '../datasets/ratings2.tab'
    data = Orange.data.Table(filename)

    data, order, shape = format_data.preprocess(data)

    # Build matrix and print it as a dense
    row = data.X[:, order[0]]
    col = data.X[:, order[1]]
    mtx = format_data.build_sparse_matrix(row, col, data.Y, shape)

    print(mtx.todense())


def create_implicit_data():
    filename = '../datasets/ratings3.tab'
    data = Orange.data.Table(filename)

    data, order, shape = format_data.preprocess(data)
    users = np.unique(data.X[:, order[0]])

    d = defaultdict(list)
    for u in users:
        indices_items = np.where(data.X[:, order[0]] == u)
        items = data.X[:, order[1]][indices_items]
        d[u] = list(items)

    for key, value in d.items():
        value2 = ' '.join(str(x) for x in value)
        print('%d\t%s' % (key, value2))
    #print(d)


if __name__ == "__main__":
    #print_dense_matrix()
    create_implicit_data()
