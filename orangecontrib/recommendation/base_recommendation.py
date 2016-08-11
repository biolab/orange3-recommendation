from Orange.base import Learner, Model
from orangecontrib.recommendation.utils import format_data

from abc import ABCMeta, abstractmethod
import numpy as np

__all__ = ["LearnerRecommendation", "ModelRecommendation"]


class ModelRecommendation(Model, metaclass=ABCMeta):

    def __init__(self):
        self.shape = (None, None)
        self.order = (0, 1)
        self.indices_missing = ([], [])

    def prepare_predict(self, X):
        # TODO: CORRECT INDICES!!! THIS IS NOT CORRECT!!!!

        # Check if all indices exist. If not, return random index.
        # On average, random indices is equivalent to return a global_average!!!
        idxs_users_missing = np.where(X[:, self.order[0]] >= self.shape[0])[0]
        idxs_items_missing = np.where(X[:, self.order[1]] >= self.shape[1])[0]

        # Check if all indices exist. If not, return 'dumb_index'.
        dumb_index = 0
        X[idxs_users_missing, self.order[0]] = dumb_index
        X[idxs_items_missing, self.order[1]] = dumb_index

        self.indices_missing = (idxs_users_missing, idxs_items_missing)

    @abstractmethod
    def predict(self): pass

    def predict_storage(self, data):
        """ Convert data.X variables to integer and calls predict(data.X)
        Args:
            data: Orange.data.Table
        Returns:
            Array with the recommendations for a given user.
        """

        # Convert indices to integer and call predict()
        data, self.order, _ = format_data.preprocess(data)
        return self.predict(data.X)

    def __str__(self):
        return self.name


class LearnerRecommendation(Learner):
    __returns__ = ModelRecommendation

    def __init__(self, preprocessors=None, verbose=False):
        self.shape = (None, None)
        self.order = (0, 1)
        self.verbose = verbose
        super().__init__(preprocessors=preprocessors)

    def prepare_fit(self, data):
        """This function calls the fit method.

        Args:
            data: Orange.data.Table

        Returns:
            Model

        """

        data, self.order, self.shape = format_data.preprocess(data)
        return data

    def prepare_model(self, model):
        model.shape = self.shape
        model.verbose = self.verbose
        return model

    @abstractmethod
    def fit_storage(self): pass

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
