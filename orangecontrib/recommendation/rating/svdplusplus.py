from orangecontrib.recommendation.rating import Learner, Model
from orangecontrib.recommendation.utils import format_data

import numpy as np
import math
from collections import defaultdict

import time
import warnings

__all__ = ['SVDPlusPlusLearner']

def _predict(u, j, global_avg, dUsers, dItems, P, Q, Y, feedback):
    bias = global_avg + dUsers[u] + dItems[j]

    # Implicit feedback
    norm_feedback = math.sqrt(len(feedback))
    if norm_feedback > 0:
        y_sum = np.sum(Y[feedback, :], axis=0)
        y_term = y_sum / norm_feedback
    else:
        y_term = 0

    # Compute base
    p_enhanced = P[u, :] + y_term
    base_pred = np.einsum('i,i', p_enhanced, Q[j, :])

    return bias + base_pred, y_term, norm_feedback


def _predict_all_items(u, global_avg, dUsers, dItems, P, Q, Y, feedback):
    bias = global_avg + dUsers[u] + dItems

    # Implicit feedback
    norm_feedback = math.sqrt(len(feedback))
    if norm_feedback > 0:
        y_sum = np.sum(Y[feedback, :], axis=0)
        y_term = y_sum / norm_feedback
    else:
        y_term = 0

    # Compute base
    p_enhanced = P[u, :] + y_term

    base_pred = np.dot(p_enhanced, Q.T)
    return bias + base_pred

def save_in_cache(matrix, key, cache):
    if key not in cache:
        if key < matrix.shape[0]:
            cache[key] = matrix.rows[key]
        else:
            cache[key] = []
    return cache[key]

def _matrix_factorization(ratings, feedback, bias, shape, order, K, steps,
                      alpha, beta,verbose=False, random_state=None):

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

        random_state:
               Random state or None.

        verbose: boolean, optional
            If true, it outputs information about the process.

    Returns:
        P (matrix, UxK), Q (matrix, KxI)

    """

    if random_state is not None:
        np.random.seed(random_state)

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

    users_cached = defaultdict(list)
    feedback_cached = defaultdict(list)

    # Factorize matrix using SGD
    for step in range(steps):
        if verbose:
            start = time.time()
            print('- Step: %d' % (step + 1))

        # Compute predictions
        for u, j in zip(*ratings.nonzero()):

            # if there is no feedback, infer it from the ratings
            if feedback is None:
                feedback_u = save_in_cache(ratings, u, users_cached)
            else:
                feedback_u = save_in_cache(feedback, u, feedback_cached)

            ruj_pred, y_term, norm_feedback = \
                _predict(u, j, globalAvg, dUsers, dItems, P, Q, Y, feedback_u)
            eij = ruj_pred - ratings[u, j]

            tempP = alpha * 2 * (eij * Q[j, :] + beta * P[u, :])
            tempQ = alpha * 2 * (eij * (P[u, :] + y_term) + beta * Q[j, :])

            if norm_feedback > 0:
                for i in feedback_u:
                    Y[i] -= alpha * 2 * ((eij/norm_feedback) * Q[j, :] +
                                         beta * Y[i])

            P[u] -= tempP
            Q[j] -= tempQ

        if verbose:
            print('\tTime: %.3fs' % (time.time() - start))

    if feedback is None:
        feedback = users_cached

    return P, Q, Y, feedback


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
                 max_rating=None, feedback=None, preprocessors=None,
                 verbose=False, random_state=None):
        self.K = K
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.P = None
        self.Q = None
        self.Y = None
        self.bias = None
        self.feedback = feedback
        self.random_state = random_state
        super().__init__(preprocessors=preprocessors, verbose=verbose,
                         min_rating=min_rating, max_rating=max_rating)

    def fit_storage(self, data):
        """This function calls the factorization method.

        Args:
            data: Orange.data.Table

        Returns:
            Model object (BRISMFModel).

        """
        data = super().prepare_fit(data)

        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well.", stacklevel=2)

        # Compute biases and global average
        self.bias = self.compute_bias(data, 'all')

        # Transform rating matrix to sparse
        data = format_data.build_sparse_matrix(data.X[:, self.order[0]],
                                               data.X[:, self.order[1]],
                                               data.Y,
                                               self.shape).tolil()
        # Factorize matrix
        self.P, self.Q, self.Y, self.feedback = \
            _matrix_factorization(ratings=data,feedback=self.feedback,
                                  bias=self.bias, shape=self.shape,
                                  order=self.order, K=self.K, steps=self.steps,
                                  alpha=self.alpha, beta=self.beta,
                                  verbose=False, random_state=self.random_state)

        model = SVDPlusPlusModel(P=self.P, Q=self.Q, Y=self.Y, bias=self.bias,
                                 feedback=self.feedback)
        return super().prepare_model(model)


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

        super().prepare_predict(X)

        users = X[:, self.order[0]]
        items = X[:, self.order[1]]

        predictions = []
        feedback_cached = defaultdict(list)
        isFeedbackADict = isinstance(self.feedback, dict)

        for i in range(0, len(users)):
            u = users[i]

            if isFeedbackADict:
                feedback_u = self.feedback[u]
            else:
                feedback_u = save_in_cache(self.feedback, u, feedback_cached)

            pred = _predict(u, items[i], self.bias['globalAvg'],
                            self.bias['dUsers'], self.bias['dItems'], self.P,
                            self.Q, self.Y, feedback_u)
            predictions.append(pred[0])

        predictions = np.asarray(predictions)
        return super().predict_on_range(predictions)

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

        predictions = []
        feedback_cached = defaultdict(list)
        isFeedbackADict = isinstance(self.feedback, dict)

        for i in range(0, len(users)):
            u = users[i]

            if isFeedbackADict:
                feedback_u = self.feedback[u]
            else:
                feedback_u = save_in_cache(self.feedback, u, feedback_cached)

            pred = _predict_all_items(u, self.bias['globalAvg'],
                                      self.bias['dUsers'], self.bias['dItems'],
                                      self.P, self.Q, self.Y, feedback_u,)
            predictions.append(pred)

        predictions = np.asarray(predictions)


        # Return top-k recommendations
        if top is not None:
            predictions = predictions[:, :top]

        return super().predict_on_range(predictions)

    def compute_objective(self, data, bias, P, Q, Y, beta):
        objective = 0

        # Transform rating matrix to sparse
        ratings = format_data.build_sparse_matrix(data.X[:, self.order[0]],
                                                  data.X[:, self.order[1]],
                                                  data.Y, self.shape).tolil()

        # Compute predictions
        for u, j in zip(*ratings.nonzero()):
            feedback_u = self.feedback[u]

            # Prediction
            b_ui = self.bias['globalAvg'] + \
                   bias['dItems'][j] + \
                   bias['dUsers'][u]

            norm_denominator = math.sqrt(len(feedback_u))
            tempN = np.sum(Y[feedback_u], axis=0)
            p_plus_y_sum_vector = tempN / norm_denominator + P[u, :]

            rij_pred = b_ui + np.dot(p_plus_y_sum_vector, Q[j, :])

            objective += (rij_pred - ratings[u, j]) ** 2

            # Regularization
            objective += beta * (np.linalg.norm(P[u, :]) ** 2
                                      + np.linalg.norm(Q[j, :]) ** 2
                                      + bias['dItems'][j] ** 2
                                      + bias['dUsers'][u] ** 2)

        # Compute RMSE
        rmse = math.sqrt(objective / float(ratings.nnz))
        return rmse

    def getPTable(self):
        variable = self.original_domain.variables[self.order[0]]
        return format_data.latent_factors_table(variable, self.P)

    def getQTable(self):
        variable = self.original_domain.variables[self.order[1]]
        return format_data.latent_factors_table(variable, self.Q)

    def getYTable(self):
        domain_name = 'Implicit feedback'
        variable = self.original_domain.variables[self.order[1]]
        return format_data.latent_factors_table(variable, self.Y, domain_name)
