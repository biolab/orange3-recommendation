from orangecontrib.recommendation.baseline import Learner, Model

import numpy as np

__all__ = ['GlobalAvgLearner']


class GlobalAvgLearner(Learner):
    """Global Average

    This model takes the average rating value of all ratings to make predictions.

    Attributes:
        verbose: boolean or int, optional
            Prints information about the process according to the verbosity
            level. Values: False (verbose=0), True (verbose=1) and INTEGER
    """

    name = 'Global average'

    def __init__(self,preprocessors=None, verbose=False):
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

        # Construct model
        model = GlobalAvgModel(global_average=np.mean(data.Y))
        return super().prepare_model(model)


class GlobalAvgModel(Model):

    def __init__(self, global_average):
        self.global_average = global_average
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

        # No need to prepare the data
        return np.full(len(X), self.global_average)

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

        # Get shape of the matrix
        num_users, num_items = self.shape

        if users is not None:
            num_users = len(users)

        # Return top-k recommendations
        if top is not None:
            num_items = top

        return np.full((num_users, num_items), self.global_average)