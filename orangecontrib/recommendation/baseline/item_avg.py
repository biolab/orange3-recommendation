from orangecontrib.recommendation.baseline import Learner, Model

import numpy as np

__all__ = ['ItemAvgLearner']

class ItemAvgLearner(Learner):
    """ Item average

    Uses the average rating value of an item to make predictions.

    Attributes:
        verbose: boolean, optional (default = False)
            Prints information about the process.
    """

    name = 'Item average'

    def __init__(self, preprocessors=None, verbose=False):
        self.bias = None
        super().__init__(preprocessors=preprocessors, verbose=verbose)

    def fit_storage(self, data):
        """This function calls the fit method.

        Args:
            data: Orange.data.Table

        Returns:
            Model object (ItemAvgModel).

        """
        data = super().prepare_fit(data)

        # Compute biases and global average
        self.bias = self.compute_bias(data, 'items')

        model = ItemAvgModel(bias=self.bias)
        return super().prepare_model(model)


class ItemAvgModel(Model):

    def __init__(self, bias):
        """This model receives a learner and provides and interface to make the
        predictions for a given user.

        Args:
            bias: dictionary
                {globalAvg: 'Global average', dItems: 'Delta items'}

       """
        self.bias = bias

    def predict(self, X):
        """This function receives an array of indexes like [[idx_item]] or
         [[idx_user, idx_item]] and returns the prediction for these pairs.

            Args:
                X: Matrix (2xN)
                    Matrix that contains pairs of the type user-item

            Returns:
                Array with the recommendations for a given user.

            """

        # Preserve just the indices of the items
        items = X[:, self.order[1]]

        predictions = self.bias['globalAvg'] + self.bias['dItems'][items]
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

        # Get shape of the matrix
        num_users, num_items = self.shape

        if users is not None:
            num_users = len(users)

        # Return top-k recommendations
        if top is not None:
            num_items = top

        tempItemsAvg = self.bias['globalAvg'] + self.bias['dItems'][:num_items]
        return np.array([tempItemsAvg,] * num_users)