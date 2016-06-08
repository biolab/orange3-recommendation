import Orange

from scipy import sparse


def tab_format():
    # Load data
    filename = '../datasets/users-movies-toy2.tab'
    data = Orange.data.Table(filename)

    for i_row in range(0, len(data.Y)):
        print('%d\t%d\t%d' % (data.X[i_row][0], data.X[i_row][1], data.Y[i_row]))


def build_sparse_matrix(self, row, col, data, shape):
    """ Given the indices of the rows, columns and its corresponding value
    this builds an sparse matrix of size 'shape'

    Args:
        row: Array of integers
           Indices of the rows for their corresponding value

        col: Array of integers
           Indices of the columns for their corresponding value

        data: Array
           Array with the values that correspond to the pair (row, col)

        shape: (int, int)
           Tuple of integers with the shape of the matrix

    Returns:
        Compressed Sparse Row matrix

    """

    mtx = sparse.csr_matrix((data, (row, col)), shape=shape)
    return mtx



def print_dense_matrix():
    # Load data
    filename = '../datasets/users-movies-toy2.tab'
    data = Orange.data.Table(filename)

    # Format data
    col_attributes = [a for a in data.domain.attributes + data.domain.metas
                      if a.attributes.get("col")]

    col_attribute = col_attributes[0] if len(
        col_attributes) == 1 else print("warning")

    row_attributes = [a for a in data.domain.attributes + data.domain.metas
                      if a.attributes.get("row")]

    row_attribute = row_attributes[0] if len(
        row_attributes) == 1 else print("warning")

    # Get indices of the columns
    idx_items = data.domain.variables.index(col_attribute)
    idx_users = data.domain.variables.index(row_attribute)

    users = len(data.domain.variables[idx_users].values)
    items = len(data.domain.variables[idx_items].values)
    shape = (users, items)

    # Build matrix and print it as a dense
    mtx = sparse.csr_matrix((data.Y, (data.X[:, 0], data.X[:, 1])), shape=shape)
    print(mtx.todense())


if __name__ == "__main__":
    print_dense_matrix()
    #tab_format()
