from Orange.base import Learner, Model

from orangecontrib.recommendation.utils import format_data

import numpy as np

__all__ = ["LearnerRecommendation", "ModelRecommendation"]


class ModelRecommendation(Model):

    def predict_storage(self, data):
        """ Convert data.X variables to integer and calls predict(data.X)

        Args:
            data: Orange.data.Table

        Returns:
            Array with the recommendations for a given user.

        """

        # Convert indices to integer and call predict()
        return self._predict(data.X.astype(int))

    def _predict(self, X):

        # Check if all indices exist. If not, return random index.
        # On average, random indices is equivalent to return a global_average!!!
        X[X[:, self.order[0]] >= self.shape[0], self.order[0]] = \
            np.random.randint(low=0, high=self.shape[0])
        X[X[:, self.order[1]] >= self.shape[1], self.order[1]] = \
            np.random.randint(low=0, high=self.shape[1])

        return self.predict_on_range(self.predict(X))


    def predict_on_range(self, predictions):
        # Just for modeling ratings with latent factors
        try:
            if self.min_rating is not None:
                predictions[predictions < self.min_rating] = self.min_rating

            if self.max_rating is not None:
                predictions[predictions > self.max_rating] = self.max_rating
        except AttributeError:
            pass
        finally:
            return predictions

    def __str__(self):
        return self.name


class LearnerRecommendation(Learner):
    __returns__ = ModelRecommendation

    def __init__(self, preprocessors=None, verbose=False):
        super().__init__(preprocessors=preprocessors)
        self.shape = (None, None)
        self.order = (0, 1)
        self.verbose = verbose

    def fit_storage(self, data):
        """This function calls the fit method.

        Args:
            data: Orange.data.Table

        Returns:
            Model

        """

        data, self.order, self.shape = format_data.preprocess(data)

        model = self.fit_model(data)
        model.shape = self.shape
        model.order = self.order
        model.verbose = self.verbose

        # Just for modeling ratings with latent factors
        try:
            model.min_rating = self.min_rating
            model.max_rating = self.max_rating
        except AttributeError:
            pass

        return model


    def compute_bias(self, data, axis='all'):
        """ Compute global average and biases of users and items

        Args:
            data: Orange.data.Table

            axis: string ('all', 'users', 'items')
                Select biases to compute

        Returns:
            bias: {globalAvg: 'Global average', dUsers: 'delta users',
            dItems: 'Delta items'}

        """

        # Compute global average
        bias = {'globalAvg': np.mean(data.Y)}

        if axis in ['all', 'users']:
            # Count non zeros in rows and columns
            # Bincount() returns an array of length np.amax(x)+1. Therefore, items
            # not rated will have a count=0. To avoid division by zero, replace
            # zeros by ones
            countings_users = np.bincount(data.X[:, self.order[0]])
            # Replace zeros by ones (Avoid problems of division by zero)
            # This only should happen during Cross-Validation!!!
            countings_users[countings_users == 0] = 1
            # Sum values along axis 0 and 1
            sums_users = np.bincount(data.X[:, self.order[0]], weights=data.Y)
            # Compute averages
            averages_users = sums_users / countings_users
            # Compute bias and deltas
            deltaUser = averages_users - bias['globalAvg']
            # Add to dictionary
            bias['dUsers'] = deltaUser


        if axis in ['all', 'items']:
            # Count non zeros in rows and columns
            # Bincount() returns an array of length np.amax(x)+1. Therefore, items
            # not rated will have a count=0. To avoid division by zero, replace
            # zeros by ones
            countings_items = np.bincount(data.X[:, self.order[1]])
            # Replace zeros by ones (Avoid problems of division by zero)
            # This only should happen during Cross-Validation!!!
            countings_items[countings_items == 0] = 1
            # Sum values along axis 0 and 1
            sums_items = np.bincount(data.X[:, self.order[1]], weights=data.Y)
            # Compute averages
            averages_items = sums_items / countings_items
            # Compute bias and deltas
            deltaItem = averages_items - bias['globalAvg']
            # Add to dictionary
            bias['dItems'] = deltaItem

        return bias
