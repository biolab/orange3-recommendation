import Orange
from orangecontrib.recommendation.utils import format_data

from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict


def tab_format():
    # Load data
    filename = 'trainingSet.csv'
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


def indices_in_range(data, min_index, max_index, max_pairs):
    logical_r_users = np.logical_and(data[:, 0] >= min_index,
                                     data[:, 0] <= max_index)
    logical_items_users = np.logical_and(data[:, 1] >= min_index,
                                         data[:, 1] <= max_index)
    indices_r_small = np.where(logical_r_users & logical_items_users)[0]

    MAX_PAIRS = 30
    if len(indices_r_small) > max_pairs:
        sample_indices = np.random.choice(len(indices_r_small), size=MAX_PAIRS,
                                          replace=False)
        indices_r_small = indices_r_small[sample_indices]
    return indices_r_small


def create_trust_data():
    MIN_INDEX = 0
    MAX_INDEX = 200
    MAX_PAIRS = 50

    # Sample ratings
    data_r = Orange.data.Table('filmtrust/ratings.tab')
    indices_r_small = indices_in_range(data_r.X, MIN_INDEX, MAX_INDEX, MAX_PAIRS)
    new_r = data_r.X[indices_r_small, :]
    print(new_r)
    data_r.X = new_r.astype(int)
    Orange.data.Table.save(data_r, "ratings_t_small.tab")

    # Sample trust
    data_t = Orange.data.Table('filmtrust/trust.tab')
    indices_t_small = indices_in_range(data_t.X, MIN_INDEX, MAX_INDEX,
                                       MAX_PAIRS)
    new_t = data_t.X[indices_t_small, :]
    print(new_t)
    data_t.X = new_t.astype(int)
    Orange.data.Table.save(data_t, 'trust_small.tab')

    asdasd = 23


if __name__ == "__main__":
    #print_dense_matrix()
    #create_implicit_data()
    create_trust_data()
