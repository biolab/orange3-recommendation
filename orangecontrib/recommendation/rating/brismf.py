from orangecontrib.recommendation.rating import Learner, Model
from orangecontrib.recommendation.utils import format_data

import numpy as np
import math

import time
import warnings

__all__ = ['BRISMFLearner']


def _predict(users, items, globalAvg, dUsers, dItems, P, Q, subscripts='i,i'):
    bias = globalAvg + dUsers[users] + dItems[items]
    base_pred = np.einsum(subscripts, P[users, :], Q[items, :])
    return bias + base_pred


def _predict_all_items(users, globalAvg, dUsers, dItems, P, Q):
    bias = globalAvg + dUsers[users]
    tempB = np.tile(np.array(dItems), (len(users), 1))
    bias = bias[:, np.newaxis] + tempB

    base_pred = np.dot(P[users], Q.T)
    return bias + base_pred


def _matrix_factorization(data, bias, shape, order, K, steps, alpha, beta,
                          verbose=False):
    """ Factorize either a dense matrix or a sparse matrix into two low-rank
        matrices which represents user and item factors.

       Args:
           data: Orange.data.Table

           K: int
               The number of latent factors.

           steps: int
               The number of epochs of stochastic gradient descent.

           alpha: float
               The learning rate of stochastic gradient descent.

           beta: float
               The regularization parameter.

           verbose: boolean, optional
               If true, it outputs information about the process.

       Returns:
           P (matrix, UxK), Q (matrix, KxI) and bias (dictionary, 'delta items'
           , 'delta users')

       """

    # Initialize factorized matrices randomly
    num_users, num_items = shape
    P = np.random.rand(num_users, K)  # User and features
    Q = np.random.rand(num_items, K)  # Item and features

    globalAvg = bias['globalAvg']
    dItems = bias['dItems']
    dUsers = bias['dUsers']

    user_col = order[0]
    item_col = order[1]
    # Factorize matrix using SGD
    for step in range(steps):
        if verbose:
            start = time.time()
            print('- Step: %d' % (step + 1))

        # Compute predictions
        for k in range(0, len(data.Y)):
            i = data.X[k][user_col]  # Users
            j = data.X[k][item_col]  # Items

            rij_pred = _predict(i, j, globalAvg, dUsers, dItems, P, Q)
            eij = rij_pred - data.Y[k]

            tempP = alpha * 2 * (eij * Q[j] + beta * P[i])
            tempQ = alpha * 2 * (eij * P[i] + beta * Q[j])
            P[i] -= tempP
            Q[j] -= tempQ

        if verbose:
            print('\tTime: %.3fs' % (time.time() - start))

    return P, Q


class BRISMFLearner(Learner):
    """ Biased Regularized Incremental Simultaneous Matrix Factorization

    This model uses stochastic gradient descent to find the values of two
    low-rank matrices which represents the user and item factors. This object
    can factorize either dense or sparse matrices.

    Attributes:
        K: int, optional
            The number of latent factors.

        steps: int, optional
            The number of epochs of stochastic gradient descent.

        alpha: float, optional
            The learning rate of stochastic gradient descent.

        beta: float, optional
            The regularization parameter.

        verbose: boolean, optional
            Prints information about the process.
    """

    name = 'BRISMF'

    def __init__(self, K=5, steps=25, alpha=0.07, beta=0.1, min_rating=None,
                 max_rating=None, preprocessors=None, verbose=False):
        self.K = K
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.P = None
        self.Q = None
        self.bias = None
        super().__init__(preprocessors=preprocessors, verbose=verbose,
                         min_rating=min_rating, max_rating=max_rating)

    def fit_model(self, data):
        """This function calls the factorization method.

        Args:
            data: Orange.data.Table

        Returns:
            Model object (BRISMFModel).

        """

        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well.", stacklevel=2)

        # Compute biases and global average
        self.bias = self.compute_bias(data, 'all')

        # Factorize matrix
        self.P, self.Q = _matrix_factorization(data=data, bias=self.bias,
                                               shape=self.shape,
                                               order=self.order, K=self.K,
                                               steps=self.steps,
                                               alpha=self.alpha,
                                               beta=self.beta, verbose=False)

        return BRISMFModel(P=self.P, Q=self.Q, bias=self.bias)


class BRISMFModel(Model):

    def __init__(self, P, Q, bias):
        """This model receives a learner and provides and interface to make the
        predictions for a given user.

        Args:
            P: Matrix (users x Latent_factors)

            Q: Matrix (items x Latent_factors)

            bias: dictionary
                {globalAvg: 'Global average', dUsers: 'delta users',
                dItems: 'Delta items'}

       """
        self.P = P
        self.Q = Q
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

        users = X[:, self.order[0]]
        items = X[:, self.order[1]]

        predictions = _predict(users, items, self.bias['globalAvg'],
                                self.bias['dUsers'], self.bias['dItems'],
                                self.P, self.Q, 'ij,ij->i')

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

        predictions = _predict_all_items(users, self.bias['globalAvg'],
                                         self.bias['dUsers'],
                                         self.bias['dItems'], self.P, self.Q)

        # Return top-k recommendations
        if top is not None:
            predictions = predictions[:, :top]

        return predictions

    def getPTable(self):
        variable = self.original_domain.variables[self.order[0]]
        return format_data.latent_factors_table(variable, self.P)

    def getQTable(self):
        variable = self.original_domain.variables[self.order[1]]
        return format_data.latent_factors_table(variable, self.Q)
