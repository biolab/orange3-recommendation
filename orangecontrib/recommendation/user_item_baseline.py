from Orange.base import Model, Learner

import numpy as np
from scipy import sparse

import time

__all__ = ['UserItemBaselineLearner']

class UserItemBaselineLearner(Learner):
    """ User-Item Baseline

    This model takes the bias of users and items plus the global average to make
    predictions.

    Attributes:
        verbose: boolean, optional (default = False)
            Prints information about the process.
    """

    name = 'User-Item Baseline'

    def __init__(self,
                 preprocessors=None,
                 verbose=False):
        self.verbose = verbose
        self.shape = None
        self.bias = None
        self.global_average = None

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
            Model object (UserItemBaselineModel).

        """

        # Optional, can be manage through preprocessors.
        data = self.format_data(data)

        # Compute bias and averages
        self.global_average = np.mean(data.Y)
        self.bias = self.compute_bias(data, self.verbose)

        return UserItemBaselineModel(bias=self.bias,
                                     global_average=self.global_average)


    def compute_bias(self, data, verbose=False):
        """ Compute averages and biases of the matrix R

        Args:
            data: Orange.data.Table

            verbose: boolean, optional
                If true, it outputs information about the process.

        Returns:
            bias (dictionary: {'delta items' , 'delta users'})

        """

        # Count non zeros in rows and columns
        countings_users = np.bincount(data.X[:, 0])
        countings_items = np.bincount(data.X[:, 1])

        # Sum values along axis 0 and 1
        sums_users = np.bincount(data.X[:, 0], weights=data.Y)
        sums_items = np.bincount(data.X[:, 1], weights=data.Y)

        # Compute averages
        averages_users = sums_users / countings_users
        averages_items = sums_items / countings_items

        # Compute bias and deltas
        deltaUser = averages_users - self.global_average
        deltaItem = averages_items - self.global_average

        bias = {'dItems': deltaItem,
                'dUsers': deltaUser}

        return bias


class UserItemBaselineModel(Model):

    def __init__(self, bias, global_average):
        """This model receives a learner and provides and interface to make the
        predictions for a given user.

        Args:
            bias: dictionary
                {'delta items', 'delta users'}

            global_average: float

       """
        self.bias = bias
        self.global_average = global_average
        self.shape = (len(bias['dUsers']), len(bias['dItems']))


    def predict(self, X):
        """This function receives an array of indexes [[idx_user, idx_item]] and
        returns the prediction for these pairs.

            Args:
                X: Matrix (2xN)
                    Matrix that contains pairs of the type user-item

            Returns:
                Array with the recommendations for a given user.

            """

        predictions = self.global_average + \
                      self.bias['dUsers'][X[:, 0]] + \
                      self.bias['dItems'][X[:, 1]]

        return predictions


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

        if users is None:
            users = np.asarray(range(0, len(self.bias['dUsers'])))

        bias = self.global_average + self.bias['dUsers'][users]
        tempB = np.tile(np.array(self.bias['dItems']), (len(users), 1))
        predictions = bias[:, np.newaxis] + tempB

        # Return top-k recommendations
        if top is not None:
            return predictions[:, :top]

        return predictions

    def __str__(self):
        return self.name