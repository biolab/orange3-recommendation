from orangecontrib.recommendation.rating import Learner, Model
from orangecontrib.recommendation.utils.format_data import *

import numpy as np
import time
import warnings

__all__ = ['BRISMFLearner']
__sparse_format__ = lil_matrix


def _predict(users, items, global_avg, dUsers, dItems, P, Q, subscripts='i,i'):
    bias = global_avg + dUsers[users] + dItems[items]
    base_pred = np.einsum(subscripts, P[users, :], Q[items, :])
    return bias + base_pred


def _predict_all_items(users, global_avg, dUsers, dItems, P, Q):
    bias = global_avg + dUsers[users]
    tempB = np.tile(np.array(dItems), (len(users), 1))
    bias = bias[:, np.newaxis] + tempB

    base_pred = np.dot(P[users], Q.T)
    return bias + base_pred


def _matrix_factorization(ratings, bias, shape, K, steps, alpha, beta,
                          verbose=False, random_state=None):

    # Seed the generator
    if random_state is not None:
        np.random.seed(random_state)

    # Get featured matrices dimensions
    num_users, num_items = shape

    # Initialize low-rank matrices
    P = np.random.rand(num_users, K)  # User-feature matrix
    Q = np.random.rand(num_items, K)  # Item-feature matrix

    # Compute bias (not need it if learnt)
    globalAvg = bias['globalAvg']
    dItems = bias['dItems']
    dUsers = bias['dUsers']

    # Factorize matrix using SGD
    for step in range(steps):
        if verbose:
            start = time.time()
            print('- Step: %d' % (step + 1))

        # Optimize rating prediction
        objective = 0
        for u, j in zip(*ratings.nonzero()):

            # Prediction and error
            rij_pred = _predict(u, j, globalAvg, dUsers, dItems, P, Q)
            eij = rij_pred - ratings[u, j]

            # Compute gradients P and Q
            tempP = eij * Q[j, :] - beta * P[u, :]
            tempQ = eij * P[u, :] - beta * Q[j, :]

            # Update the gradients at the same time
            # I use the loss function divided by 2, to simplify the gradients
            P[u, :] -= alpha * tempP
            Q[j, :] -= alpha * tempQ

            # Loss function
            if verbose:
                objective += eij ** 2
                objective += beta * (bias['dUsers'][u] ** 2 +
                                     bias['dItems'][j] ** 2 +
                                     np.linalg.norm(P[u, :]) ** 2
                                     + np.linalg.norm(Q[j, :]) ** 2)

        # Loss function (Remember it must be divided by 2 to be correct)
        if verbose:
            print('\tLoss: %.3f' % (objective*0.5))
            print('\tTime: %.3fs' % (time.time() - start))
            print('')

    return P, Q


class BRISMFLearner(Learner):
    """BRISMF: Biased Regularized Incremental Simultaneous Matrix Factorization

    This model uses stochastic gradient descent to find two low-rank
    matrices: user-feature matrix and item-feature matrix.

    Attributes:
        K: int, optional
            The number of latent factors.

        steps: int, optional
            The number of passes over the training data (aka epochs).

        alpha: float, optional
            The learning rate.

        beta: float, optional
            The regularization for the ratings.

        min_rating: float, optional
            Defines the lower bound for the predictions. If None (default),
            ratings won't be bounded.

        max_rating: float, optional
            Defines the upper bound for the predictions. If None (default),
            ratings won't be bounded.

        verbose: boolean, optional
            Prints information about the process.

        random_state: int, optional
            Set the seed for the numpy random generator, so it makes the random
            numbers predictable. This a debbuging feature.

    """

    name = 'BRISMF'

    def __init__(self, K=5, steps=25, alpha=0.07, beta=0.1, min_rating=None,
                 max_rating=None, preprocessors=None, verbose=False,
                 random_state=None):
        self.K = K
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state
        super().__init__(preprocessors=preprocessors, verbose=verbose,
                         min_rating=min_rating, max_rating=max_rating)

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

        # Check convergence
        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well.", stacklevel=2)

        # Compute biases (not need it if learnt)
        bias = self.compute_bias(data, 'all')

        # Transform ratings matrix into a sparse matrix
        data = table2sparse(data, self.shape, self.order,
                            type=__sparse_format__)

        # Factorize matrix
        P, Q = _matrix_factorization(ratings=data, bias=bias, shape=self.shape,
                                     K=self.K, steps=self.steps,
                                     alpha=self.alpha, beta=self.beta,
                                     verbose=self.verbose,
                                     random_state=self.random_state)

        model = BRISMFModel(P=P, Q=Q, bias=bias)
        return super().prepare_model(model)


class BRISMFModel(Model):
    def __init__(self, P, Q, bias):
        self.P = P
        self.Q = Q
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

        predictions = _predict(users, items, self.bias['globalAvg'],
                               self.bias['dUsers'], self.bias['dItems'],
                               self.P, self.Q, 'ij,ij->i')

        # Set predictions for non-existing indices (CV)
        predictions = self.fix_predictions(X, predictions, self.bias)
        return super().predict_on_range(predictions)

    def predict_items(self, users=None, top=None):
        """This function returns all the predictions for a set of items.

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

        predictions = _predict_all_items(users, self.bias['globalAvg'],
                                         self.bias['dUsers'],
                                         self.bias['dItems'], self.P, self.Q)

        # Return top-k recommendations
        if top is not None:
            predictions = predictions[:, :top]

        return super().predict_on_range(predictions)

    def compute_objective(self, data, beta):
        # TODO: Cast rows, cols through preprocess
        data.X = data.X.astype(int)  # Convert indices to integer

        users = data.X[:, self.order[0]]
        items = data.X[:, self.order[1]]

        globalAvg = self.bias['globalAvg']
        dItems = self.bias['dItems']
        dUsers = self.bias['dUsers']

        objective = 0
        subscripts = 'i,i'

        if len(users) > 1:
            subscripts = 'ij,ij->i'
        predictions = _predict(users, items, globalAvg, dUsers, dItems, self.P,
                               self.Q, subscripts)
        objective += (predictions - data.Y) ** 2

        # Regularization
        objective += beta * (np.linalg.norm(self.P[users, :], axis=1) ** 2
                             + np.linalg.norm(self.Q[items, :], axis=1) ** 2
                             + dItems[items] ** 2 + dUsers[users] ** 2)
        return objective.sum()

    def getPTable(self):
        variable = self.original_domain.variables[self.order[0]]
        return feature_matrix(variable, self.P)

    def getQTable(self):
        variable = self.original_domain.variables[self.order[1]]
        return feature_matrix(variable, self.Q)
