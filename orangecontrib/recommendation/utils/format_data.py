import Orange

from scipy import sparse
import numpy as np

def preprocess(data):
    """Transforms the raw data read by Orange into something that this
    class can use.
    data.X are converted to integers (it is use as indices), order are the
    index of users and items in the file (order[0]=idx_users;
    order[1]=idx_items), shape (total users, total items).

    Args:
        data: Orange.data.Table

    Returns:
        data: Orange.data.Table
        order: (int, int)
        shape: (int, int)

    """
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

    users = int(np.max(data.X[:, idx_users]) + 1)
    items = int(np.max(data.X[:, idx_items]) + 1)

    order = (idx_users, idx_items)
    shape = (users, items)

    # Convert to integer
    data.X = data.X.astype(int)

    return data, order, shape


def build_sparse_matrix(row, col, values, shape):
    """ Given the indices of the rows, columns and its corresponding value
    this builds an sparse matrix of size 'shape'

    Args:
        row: Array of integers
           Indices of the rows for their corresponding value

        col: Array of integers
           Indices of the columns for their corresponding value

        values: Array
           Array with the values that correspond to the pair (row, col)

        shape: (int, int)
           Tuple of integers with the shape of the matrix

    Returns:
        Compressed Sparse Row matrix

    """

    return sparse.csr_matrix((values, (row, col)), shape=shape)