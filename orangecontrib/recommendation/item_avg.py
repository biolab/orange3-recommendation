from Orange.base import Model, Learner

import numpy as np
from scipy import sparse

__all__ = ['ItemAvgLearner']

class ItemAvgLearner(Learner):
    """ Item average

    Uses the average rating value of an item to make predictions.

    Attributes:
        verbose: boolean, optional (default = False)
            Prints information about the process.
    """

    name = 'Item average'

    def __init__(self,
                 preprocessors=None,
                 verbose=False):
        self.verbose = verbose
        self.shape = None

        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def format_data(self, data):
        """Transforms the raw data read by Orange into something that this
        class can use

        Args:
            data: Orange.data.Table

        Returns:
            data

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

        users = len(data.domain.variables[idx_users].values)
        items = len(data.domain.variables[idx_items].values)
        self.shape = (users, items)

        # Convert to integer
        data.X = data.X.astype(int)

        return data


    def fit_storage(self, data):
        """This function calls the factorization method.

        Args:
            data: Orange.data.Table

        Returns:
            Model object (ItemAvgModel).

        """

        # Optional, can be manage through preprocessors.
        data = self.format_data(data)

        # build sparse matrix
        R = self.build_sparse_matrix(data.X[:, 0],
                                     data.X[:, 1],
                                     data.Y,
                                     self.shape)

        return ItemAvgModel(items_average=np.ravel(R.mean(axis=0)),
                            shape=self.shape)


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


class ItemAvgModel(Model):

    def __init__(self, items_average, shape):
        """This model receives a learner and provides and interface to make the
        predictions for a given user.

        Args:
            items_average: Array
            shape: (int, int)

       """
        self.items_average = items_average
        self.shape = shape


    def predict(self, X):
        """This function receives an array of indexes like [[idx_item]] or
         [[idx_user, idx_item]] and returns the prediction for these pairs.

            Args:
                X: Matrix (2xN)
                    Matrix that contains pairs of the type user-item

            Returns:
                Array with the recommendations for a given user.

            """

        if X.shape[1] > 1:
            X = X[:, 1]

        return self.items_average[X]


    def predict_storage(self, data):
        """ Convert data.X variables to integer and calls predict(data.X)

        Args:
            data: Orange.data.Table

        Returns:
            Array with the recommendations for a given user.

        """

        # Convert indices to integer and call predict()
        return self.predict(data.X.astype(int))


    def predict_items(self, users=None, top=None):
        """This function returns all the predictions for a set of items.
        If users is set to 'None', it will return all the predictions for all
        the users (matrix of size [num_users x num_items]).

        Args:
            users: array, optional
                Array with the indices of the users to which make the
                predictions.

            top: int, optional
                Return just the first k recommendations.

        Returns:
            Array with the recommendations for requested users.

        """

        # Get shape of the matrix
        num_users, num_items = self.shape

        if users is not None:
            num_users = len(users)

        # Return top-k recommendations
        if top is not None:
            num_items = top

        tempItemsAvg = self.items_average[:num_items]
        return np.array([tempItemsAvg,] * num_users)


    def __str__(self):
        return self.name