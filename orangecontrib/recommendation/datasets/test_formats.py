import Orange
from orangecontrib.recommendation.utils import format_data


def tab_format():
    # Load data
    filename = '../datasets/users-movies-toy2.tab'
    data = Orange.data.Table(filename)

    for i_row in range(0, len(data.Y)):
        print('%d\t%d\t%d' % (data.X[i_row][0], data.X[i_row][1], data.Y[i_row]))


def print_dense_matrix():
    # Load data
    filename = '../datasets/users-movies-toy2.tab'
    data = Orange.data.Table(filename)

    data, order, shape = format_data.format_data(data)

    # Build matrix and print it as a dense
    row = data.X[:, order[0]]
    col = data.X[:, order[1]]
    mtx = format_data.build_sparse_matrix(row, col, data.Y, shape)

    print(mtx.todense())


if __name__ == "__main__":
    print_dense_matrix()
    #tab_format()
