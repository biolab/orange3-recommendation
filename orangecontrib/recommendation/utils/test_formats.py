import Orange
from orangecontrib.recommendation.utils import format_data

from scipy.sparse import csr_matrix

def tab_format():
    # Load data
    filename = '/Users/salvacarrion/Desktop/data_binary.tab'
    data = Orange.data.Table(filename)

    A = csr_matrix(data.X)

    indices = A.nonzero()
    for i in range(0, len(indices[0])):
        print('%d\t%d\t%d' % (indices[0][i], indices[1][i], 1))


def print_dense_matrix():
    # Load data
    filename = '../datasets/ratings2.tab'
    data = Orange.data.Table(filename)

    data, order, shape = format_data.format_data(data)

    # Build matrix and print it as a dense
    row = data.X[:, order[0]]
    col = data.X[:, order[1]]
    mtx = format_data.build_sparse_matrix(row, col, data.Y, shape)

    print(mtx.todense())


if __name__ == "__main__":
    #print_dense_matrix()
    tab_format()
