from orangecontrib.recommendation.baseline import Learner, Model

import numpy as np

__all__ = ['UserAvgLearner']

class UserAvgLearner(Learner):
    """ User average

    Uses the average rating value of a user to make predictions.

    Attributes:
        verbose: boolean, optional (default = False)
            Prints information about the process.
    """

    name = 'User average'

    def __init__(self, preprocessors=None, verbose=False):
        self.bias = None
        super().__init__(preprocessors=preprocessors, verbose=verbose)

    def fit_model(self, data):
        """This function calls the fit method.

        Args:
            data: Orange.data.Table

        Returns:
            Model object (UserAvgModel).

        """

        # Compute biases and global average
        self.bias = self.compute_bias(data, 'users')
        return UserAvgModel(bias=self.bias)


class UserAvgModel(Model):

    def __init__(self, bias):
        """This model receives a learner and provides and interface to make the
        predictions for a given user.

        Args:
            bias: dictionary
                {globalAvg: 'Global average', dUsers: 'Delta users'}
       """
        self.bias = bias

    def predict(self, X):
        """This function receives an array of indexes like [[idx_user]] or
         [[idx_user, idx_item]] and returns the prediction for these pairs.

            Args:
                X: Matrix (mxn),
                    Matrix that contains pairs of the type user-item

            Returns:
                Array with the recommendations for a given user.

            """

        # Preserve just the indices of the items
        users = X[:, self.order[0]]

        predictions = self.bias['globalAvg'] + self.bias['dUsers'][users]
        return predictions

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

        predictions = self.bias['globalAvg'] + self.bias['dUsers'][users]
        predictions = np.tile(predictions[:, np.newaxis], (1, top))

        return predictions