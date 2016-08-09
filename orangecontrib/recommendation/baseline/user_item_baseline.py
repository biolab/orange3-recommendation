from orangecontrib.recommendation.baseline import Learner, Model

import numpy as np

__all__ = ['UserItemBaselineLearner']


class UserItemBaselineLearner(Learner):
    """User-Item baseline

    This model takes the bias of users and items plus the global average to make
    predictions.

    Attributes:
        verbose: boolean or int, optional
            Prints information about the process according to the verbosity
            level. Values: False (verbose=0), True (verbose=1) and INTEGER
    """

    name = 'User-Item Baseline'

    def __init__(self, preprocessors=None, verbose=False):
        super().__init__(preprocessors=preprocessors, verbose=verbose)

    def fit_storage(self, data):
        """Fit the model according to the given training data.

        Args:
            data: Orange.data.Table

        Returns:
            self: object
                Returns self.

        """

        # Prepare data
        data = super().prepare_fit(data)

        # Compute biases and global average
        bias = self.compute_bias(data, 'all')

        # Construct model
        model = UserItemBaselineModel(bias=bias)
        return super().prepare_model(model)


class UserItemBaselineModel(Model):

    def __init__(self, bias):
        self.bias = bias
        super().__init__()

    def predict(self, X):
        """Perform predictions on samples in X.

        This function receives an array of indices and returns the prediction
        for each one.

        Args:
            X: ndarray
                Samples. Matrix that contains user-item pairs.

        Returns:
            C: array, shape = (n_samples,)
                Returns predicted values.

        """

        # Prepare data (set valid indices for non-existing (CV))
        super().prepare_predict(X)

        users = X[:, self.order[0]]
        items = X[:, self.order[1]]

        predictions = self.bias['globalAvg'] + self.bias['dUsers'][users] + \
                      self.bias['dItems'][items]

        # Set predictions for non-existing indices (CV)
        predictions = self.fix_predictions(X, predictions, self.bias)
        return predictions

    def predict_items(self, users=None, top=None):
        """Perform predictions on samples in 'users' for all items.

        Args:
            users: array, optional
                Array with the indices of the users to which make the
                predictions. If None (default), predicts for all users.

            top: int, optional
                Returns the k-first predictions. (Do not confuse with
                'top-best').

        Returns:
            C: ndarray, shape = (n_samples, n_items)
                Returns predicted values.

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
