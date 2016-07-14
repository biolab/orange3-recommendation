from orangecontrib.recommendation.baseline import Learner, Model

import numpy as np

__all__ = ['GlobalAvgLearner']


class GlobalAvgLearner(Learner):
    """ Global Average

    Uses the average rating value of all ratings to make predictions.

    Attributes:

        verbose: boolean, optional (default = False)
            Prints information about the process.
    """

    name = 'Global average'

    def __init__(self,preprocessors=None, verbose=False):
        super().__init__(preprocessors=preprocessors, verbose=verbose)

    def fit_model(self, data):
        """This function calls the fit method.

        Args:
            data: Orange.data.Table

        Returns:
            Model object (GlobalAvgModel).

        """

        return GlobalAvgModel(global_average=np.mean(data.Y))


class GlobalAvgModel(Model):

    def __init__(self, global_average):
        """This model receives a learner and provides and interface to make the
        predictions for a given user.

        Args:
            global_average: float

       """
        self.global_average = global_average

    def predict(self, X):
        """This function receives an array of indexes [[idx_user, idx_item]] and
        returns the prediction for these pairs.

            Args:
                X: Matrix (2xN)
                    Matrix that contains pairs of the type user-item

            Returns:
                Array with the recommendations for a given user.

            """

        return np.full(len(X), self.global_average)

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

        return np.full((num_users, num_items), self.global_average)