from orangecontrib.recommendation.baseline import Learner, Model

import numpy as np

__all__ = ['UserAvgLearner']

class UserAvgLearner(Learner):
    """User average

    This model takes the average rating value of a user to make predictions.

    Attributes:
        verbose: boolean or int, optional
            Prints information about the process according to the verbosity
            level. Values: False (verbose=0), True (verbose=1) and INTEGER
    """

    name = 'User average'

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
        bias = self.compute_bias(data, 'users')

        # Construct model
        model = UserAvgModel(bias=bias)
        return super().prepare_model(model)


class UserAvgModel(Model):

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

        # Preserve just the indices of the items
        users = X[:, self.order[0]]

        predictions = self.bias['globalAvg'] + self.bias['dUsers'][users]

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
            users = np.asarray(range(0, self.shape[0]))

        # Return top-k recommendations
        if top is None:
            top = self.shape[1]

        predictions = self.bias['globalAvg'] + self.bias['dUsers'][users]
        predictions = np.tile(predictions[:, np.newaxis], (1, top))

        return predictions