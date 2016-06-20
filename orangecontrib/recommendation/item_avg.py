from orangecontrib.recommendation import Learner, Model
from orangecontrib.recommendation.utils import format_data

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
        self.order = None

        super().__init__(preprocessors=preprocessors)
        self.params = vars()


    def fit_storage(self, data):
        """This function calls the factorization method.

        Args:
            data: Orange.data.Table

        Returns:
            Model object (ItemAvgModel).

        """

        # Optional, can be manage through preprocessors.
        data, self.order, self.shape = format_data.preprocess(data)

        # Compute averages
        averages_items = self.compute_averages(data)

        return ItemAvgModel(items_average=averages_items,
                            shape=self.shape,
                            order=self.order)


    def compute_averages(self, data):
        """This function computes the averages of the users

        Args:
            data: Orange.data.Table

        Returns:
            Array

        """

        # Count non zeros in rows
        # Bincount() returns an array of length np.amax(x)+1. Therefore, items
        # not rated will have a count=0. To avoid division by zero, replace
        # zeros by ones
        countings_items = np.bincount(data.X[:, self.order[1]])

        # Replace zeros by ones (Avoid problems of division by zero)
        # This only should happen during Cross-Validation
        countings_items[countings_items == 0] = 1

        # Sum values along axis 1
        sums_items = np.bincount(data.X[:, self.order[1]], weights=data.Y)

        # Compute averages
        averages_items = sums_items / countings_items

        return averages_items


class ItemAvgModel(Model):

    def __init__(self, items_average, shape, order):
        """This model receives a learner and provides and interface to make the
        predictions for a given user.

        Args:
            items_average: Array

            shape: (int, int)

            order: (int, int)
                Tuple with the index of the columns users and items in X. (idx_user, idx_item)

       """
        self.items_average = items_average
        self.shape = shape
        self.order = order


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
            X = X[:, self.order[1]]

        # Check if all indices exist. If not, return random index.
        # On average, random indices is equivalent to return a global_average
        X[X >= self.shape[1]] = np.random.randint(low=0, high=self.shape[1])

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