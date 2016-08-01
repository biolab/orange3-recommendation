from Orange.data import Table, Domain, ContinuousVariable, StringVariable

import numpy as np
from scipy import sparse


def preprocess(data):
    """ Preprocess the raw data read by Orange3 into something that can be used
    in this module.

    'data.X' is casted to integers (since it contains the indices of the
    data), 'order' is a tuple with the indices of the columns users and items
    in the original file, as well as in 'data.X' matrix.

    Note:
        I talk about 'Users' and 'Items' but this can be whatever pair of
        elements you want like trusters-trustees, team1-team2,...

        Clarification:
             - order: (idx_users; idx_items)
             - shape: (total users; total items).

    Args:
        data: Orange.data.Table

    Returns:
        data: Orange.data.Table
        order: (int, int)
        shape: (int, int)

    """

    # Get index of column with the 'col' attribute set to 1 (col=1)
    col_attributes = [a for a in data.domain.attributes + data.domain.metas
                      if a.attributes.get("col")]
    col_attribute = col_attributes[0] if len(
        col_attributes) == 1 else print("warning")

    # Get index of column with the 'row' attribute set to 1 (row=1)
    row_attributes = [a for a in data.domain.attributes + data.domain.metas
                      if a.attributes.get("row")]
    row_attribute = row_attributes[0] if len(
        row_attributes) == 1 else print("warning")

    # Get indices of the columns
    idx_items = data.domain.variables.index(col_attribute)
    idx_users = data.domain.variables.index(row_attribute)

    # Find the highest value on each column
    users = int(np.max(data.X[:, idx_users]) + 1)
    items = int(np.max(data.X[:, idx_items]) + 1)

    # Construct tuples
    order = (idx_users, idx_items)
    shape = (users, items)

    # Cast indices to integer
    data.X = data.X.astype(int)

    return data, order, shape


def sparse_matrix_2d(row, col, data, shape):
    """ Constructs a 2D sparse matrix.

    Given the indices of the rows, columns and value
    this constructs a sparse matrix of size 'shape'

    Args:
        row: Array of integers
           Row indices

        col: Array of integers
           Column indices

        data: Array
           Array with the values mapped to the pair (row, col)

        shape: (int, int)
           Tuple of integers with the shape of the matrix

    Returns:
        Compressed Sparse Row matrix

    """

    return sparse.csr_matrix((data, (row, col)), shape=shape)


def feature_matrix(variable, matrix, domain_name=None):
    factors_name = [ContinuousVariable('K' + str(i + 1))
                     for i in range(len(matrix[0, :]))]

    if domain_name is None:
        dname = variable.name
    else:
        dname = domain_name

    if isinstance(variable, ContinuousVariable):
        domain_val = ContinuousVariable(dname)
        values = np.atleast_2d(np.arange(0, len(matrix))).T
    else:
        domain_val = StringVariable(dname)
        values = np.column_stack((variable.values,))

    tDomain = Domain(factors_name, None, [domain_val])
    return Table(tDomain, matrix, None, values)
