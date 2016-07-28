from orangecontrib.recommendation.baseline import Learner, Model

import numpy as np

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

    def __init__(self, preprocessors=None, verbose=False):
        self.bias = None
        super().__init__(preprocessors=preprocessors, verbose=verbose)

    def fit_storage(self, data):
        """This function calls the fit method.

        Args:
            data: Orange.data.Table

        Returns:
            Model object (UserItemBaselineModel).

        """
        data = super().prepare_fit(data)

        # Compute biases and global average
        self.bias = self.compute_bias(data, 'all')

        model = UserItemBaselineModel(bias=self.bias)
        return super().prepare_model(model)


class UserItemBaselineModel(Model):

    def __init__(self, bias):
        """This model receives a learner and provides and interface to make the
        predictions for a given user.

        Args:
            bias: dictionary
                {globalAvg: 'Global average', dUsers: 'delta users',
                dItems: 'Delta items'}

       """
        self.bias = bias

    def predict(self, X):
        """This function receives an array of indexes [[idx_user, idx_item]] and
        returns the prediction for these pairs.

            Args:
                X: Matrix (2xN)
                    Matrix that contains pairs of the type user-item

            Returns:
                Array with the recommendations for a given user.

            """

        super().prepare_predict(X)

        users = X[:, self.order[0]]
        items = X[:, self.order[1]]

        predictions = self.bias['globalAvg'] + self.bias['dUsers'][users] + \
                      self.bias['dItems'][items]
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
            users = np.asarray(range(0, len(self.bias['dUsers'])))

        bias = self.bias['globalAvg'] + self.bias['dUsers'][users]
        tempB = np.tile(np.array(self.bias['dItems']), (len(users), 1))
        predictions = bias[:, np.newaxis] + tempB

        # Return top-k recommendations
        if top is not None:
            predictions = predictions[:, :top]

        return predictions
