from orangecontrib.recommendation.rating import Learner, Model
from orangecontrib.recommendation.utils import format_data

import numpy as np
import math

import time
import warnings

__all__ = ['SVDPlusPlusLearner']


def _predict(i, j, globalAvg, dUsers, dItems, P, Q, Y, feedback):
    b_ui = globalAvg + dUsers[i] + dItems[j]

    norm_denominator = math.sqrt(len(feedback))
    tempN = np.sum(Y[feedback], axis=0)
    p_plus_y_sum_vector = tempN / norm_denominator + P[i, :]

    rij_pred = b_ui + np.dot(p_plus_y_sum_vector, Q[j, :])

    return rij_pred, p_plus_y_sum_vector, norm_denominator


def _predict2(users, items, globalAvg, dUsers, dItems, P, Q, Y, feedback):
    bias = globalAvg + dUsers[users] + dItems[items]

    base_pred = []
    for k in range(len(users)):
        i = users[k]
        j = items[k]
        f = feedback[i]  # Implicit data

        norm_denominator = np.sqrt(len(f))
        tempN = np.sum(Y[f], axis=0)

        p_plus_y_sum_vector = tempN / norm_denominator + P[i, :]
        base_pred.append(np.dot(p_plus_y_sum_vector, Q[j, :]))

    predictions = bias + np.asarray(base_pred)
    return predictions


def _predict3(users, globalAvg, dUsers, dItems, P, Q, Y, feedback):
    bias = globalAvg + dUsers[users]
    tempB = np.tile(np.array(dItems), (len(users), 1))
    bias = bias[:, np.newaxis] + tempB

    predictions = []
    for i in range(len(users)):
        u = users[i]
        f = feedback[u]  # Implicit data

        norm_denominator = np.sqrt(len(f))
        tempN = np.sum(Y[f], axis=0)

        p_plus_y_sum_vector = tempN / norm_denominator + P[u, :]
        pred = bias[i, :]
        pred2 = np.dot(p_plus_y_sum_vector, Q.T)

        predictions.append(pred + pred2)

    predictions = np.asarray(predictions)
    return predictions


def _matrix_factorization(data, bias, feedback, shape, order, K, steps,
                          alpha, beta, verbose=False):

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
        P (matrix, UxK), Q (matrix, KxI)

    """

    # Initialize factorized matrices randomly
    num_users, num_items = shape
    P = np.random.rand(num_users, K)  # User and features
    Q = np.random.rand(num_items, K)  # Item and features
    Y = np.random.randn(num_items, K)

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
            i = data.X[k][user_col]  # User
            j = data.X[k][item_col]  # Item
            f = feedback[i]  # Implicit data

            rij_pred, p_plus_y_sum_vector, norm_denominator = \
                _predict(i, j, globalAvg, dUsers, dItems, P, Q, Y, f)
            eij = rij_pred - data.Y[k]

            tempP = alpha * 2 * (eij * Q[j] + beta * P[i])
            tempQ = alpha * 2 * (eij * p_plus_y_sum_vector + beta * Q[j])
            tempY = alpha * 2 * (eij / norm_denominator * Q[j] + beta * Y[f])

            Q[j] -= tempQ
            P[i] -= tempP
            Y[f] -= tempY

        if verbose:
            print('\tTime: %.3fs' % (time.time() - start))

    return P, Q, Y


class SVDPlusPlusLearner(Learner):
    """ SVD++: Matrix factorization that also takes into account what users have
    rated

    Attributes:
        K: int, optional
            The number of latent factors.

        steps: int, optional (default = 100)
            The number of epochs of stochastic gradient descent.

        alpha: float, optional (default = 0.005)
            The learning rate of stochastic gradient descent.

        beta: float, optional (default = 0.02)
            The regularization parameter.

        verbose: boolean, optional (default = False)
            Prints information about the process.
    """

    name = 'SVD++'

    def __init__(self, K=5, steps=25, alpha=0.07, beta=0.1, min_rating=None,
                 max_rating=None, preprocessors=None, verbose=False):
        self.K = K
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.P = None
        self.Q = None
        self.Y = None
        self.bias = None
        self.feedback = None
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

        # Generate implicit feedback using the explicit ratings of the data
        self.feedback = self.generate_implicit_feedback(data)

        # Factorize matrix
        self.P, self.Q, self.Y =\
            _matrix_factorization(data=data, bias=self.bias,
                                  feedback=self.feedback, shape=self.shape,
                                  order=self.order, K=self.K, steps=self.steps,
                                  alpha=self.alpha, beta=self.beta,
                                  verbose=False)

        return SVDPlusPlusModel(P=self.P, Q=self.Q, Y=self.Y, bias=self.bias,
                                feedback=self.feedback)

    def generate_implicit_feedback(self, data):
        users = np.unique(data.X[:, self.order[0]])
        feedback = {}

        for u in users:
            indices_items = np.where(data.X[:, self.order[0]] == u)
            items = data.X[:, self.order[1]][indices_items]
            feedback[u] = items
        return feedback


class SVDPlusPlusModel(Model):

    def __init__(self, P, Q, Y, bias, feedback):
        """This model receives a learner and provides and interface to make the
        predictions for a given user.

        Args:
            P: Matrix (users x Latent_factors)

            Q: Matrix (items x Latent_factors)

            bias: dictionary
                {'delta items', 'delta users'}

            feedback: dictionary
                {user_id: [ratings]}

       """
        self.P = P
        self.Q = Q
        self.Y = Y
        self.bias = bias
        self.feedback = feedback

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

        predictions = _predict2(users, items, self.bias['globalAvg'],
                                self.bias['dUsers'], self.bias['dItems'],
                                self.P, self.Q, self.Y, self.feedback)

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

        predictions = _predict3(users, self.bias['globalAvg'],
                                self.bias['dUsers'], self.bias['dItems'],
                                self.P, self.Q, self.Y, self.feedback)

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

    def getYTable(self):
        variable = self.original_domain.variables[self.order[1]]
        return format_data.latent_factors_table(variable, self.Y)
