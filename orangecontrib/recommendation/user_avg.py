from Orange.base import Model, Learner

import numpy as np
from scipy import sparse

__all__ = ['UserAvgLearner']

class UserAvgLearner(Learner):
    """ User average

    Uses the average rating value of a user to make predictions.

    Attributes:
        verbose: boolean, optional (default = False)
            Prints information about the process.
    """

    name = 'User average'

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
            Model object (UserAvgModel).

        """

        # Optional, can be manage through preprocessors.
        data = self.format_data(data)

        # Compute averages
        averages_users = self.compute_averages(data)

        return UserAvgModel(users_average=averages_users,
                            shape=self.shape)


    def compute_averages(self, data):
        """This function computes the averages of the items

        Args:
            data: Orange.data.Table

        Returns:
            Array

        """

        # Count non zeros in columns
        countings_users = np.bincount(data.X[:, 0])

        # Sum values along axis 0
        sums_users = np.bincount(data.X[:, 0], weights=data.Y)

        # Compute averages
        averages_users = sums_users / countings_users

        return averages_users


class UserAvgModel(Model):

    def __init__(self, users_average, shape):
        """This model receives a learner and provides and interface to make the
        predictions for a given user.

        Args:
            users_average: Array
            shape: (int, int)

       """
        self.users_average = users_average
        self.shape = shape


    def predict(self, X):
        """This function receives an array of indexes like [[idx_user]] or
         [[idx_user, idx_item]] and returns the prediction for these pairs.

            Args:
                X: Matrix (mxn),
                    Matrix that contains pairs of the type user-item

            Returns:
                Array with the recommendations for a given user.

            """

        if X.shape[1] > 1:
            X = X[:, 0]

        return self.users_average[X]


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
            users = np.asarray(range(0, self.shape[0]))

        # Return top-k recommendations
        if top is None:
            top = self.shape[1]

        predictions = self.users_average[users]
        predictions = np.tile(predictions[:, np.newaxis], (1, top))

        return predictions


    def __str__(self):
        return self.name