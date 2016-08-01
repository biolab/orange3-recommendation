from orangecontrib.recommendation.rating import Learner, Model
from orangecontrib.recommendation.utils.format_data import *

from collections import defaultdict

import numpy as np
import math
import time
import warnings

__all__ = ['SVDPlusPlusLearner']
__sparse_format__ = lil_matrix


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
    res = cache.get(key)
    if res is None:
        if key < matrix.shape[0]:
            res = np.asarray(matrix.rows[key])
        else:
            res = []
        cache[key] = res
    return res


def _matrix_factorization(ratings, feedback, bias, shape, K, steps,
                      alpha, beta,verbose=False, random_state=None):

    # Seed the generator
    if random_state is not None:
        np.random.seed(random_state)

    # Get featured matrices dimensions
    num_users, num_items = shape

    # Initialize low-rank matrices
    P = np.random.rand(num_users, K)  # User-feature matrix
    Q = np.random.rand(num_items, K)  # Item-feature matrix
    Y = np.random.randn(num_items, K) # Feedback-feature matrix

    # Compute bias (not need it if learnt)
    globalAvg = bias['globalAvg']
    dItems = bias['dItems']
    dUsers = bias['dUsers']

    # Cache rows
    users_cached = defaultdict(list)
    feedback_cached = defaultdict(list)

    # Factorize matrix using SGD
    for step in range(steps):
        if verbose:
            start = time.time()
            print('- Step: %d' % (step + 1))

        # Optimize rating prediction
        objective = 0
        for u, j in zip(*ratings.nonzero()):

            # if there is no feedback, infer it from the ratings
            if feedback is None:
                feedback_u = save_in_cache(ratings, u, users_cached)
            else:
                feedback_u = save_in_cache(feedback, u, feedback_cached)
                feedback_u = feedback_u[feedback_u < num_items]  # For CV

            # Prediction and error
            ruj_pred, y_term, norm_feedback = \
                _predict(u, j, globalAvg, dUsers, dItems, P, Q, Y, feedback_u)
            eij = ruj_pred - ratings[u, j]

            # Gradients of P and Q
            tempP = alpha * -2 * (eij * Q[j, :] - beta * P[u, :])
            tempQ = alpha * -2 * (eij * (P[u, :] + y_term) - beta * Q[j, :])

            # Gradient Y
            if norm_feedback > 0:
                for i in feedback_u:
                    Y[i, :] += alpha * -2 * (eij/norm_feedback * Q[j, :]
                                            - beta * Y[i, :])

            # Update the gradients at the same time
            P[u] += tempP
            Q[j] += tempQ

            # Loss function
            if verbose:
                objective += eij ** 2
                temp_y = np.sum(Y[feedback_u, :], axis=0)
                objective += beta * (bias['dUsers'][u] ** 2 +
                                     bias['dItems'][j] ** 2 +
                                     np.linalg.norm(P[u, :]) ** 2
                                     + np.linalg.norm(Q[j, :]) ** 2
                                     + np.linalg.norm(temp_y) ** 2)

        if verbose:
            print('\t- Loss: %.3f' % objective)
            print('\t- Time: %.3fs' % (time.time() - start))
            print('')

    if feedback is None:
        feedback = users_cached

    return P, Q, Y, feedback


class SVDPlusPlusLearner(Learner):
    """SVD++ matrix factorization

    This model uses stochastic gradient descent to find three low-rank
    matrices: user-feature matrix, item-feature matrix and feedback-feature
    matrix.

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

        feedback: Orange.data.Table
            Implicit feedback information. If None (default), implicit
            information will be inferred from the ratings (e.g.: item rated,
            means items seen).

        verbose: boolean, optional
            Prints information about the process.

        random_state: int, optional
            Set the seed for the numpy random generator, so it makes the random
            numbers predictable. This a debbuging feature.

    """

    name = 'SVD++'

    def __init__(self, K=5, steps=25, alpha=0.07, beta=0.1, min_rating=None,
                 max_rating=None, feedback=None, preprocessors=None,
                 verbose=False, random_state=None):
        self.K = K
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state

        self.feedback = feedback
        if feedback is not None:
            self.feedback, order_f, self.shape_f = preprocess(feedback)

            # Transform feedback matrix into a sparse matrix
            self.feedback = table2sparse(self.feedback, self.shape_f,
                                         order_f, type=__sparse_format__)

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
        P, Q, Y, new_feedback = \
            _matrix_factorization(ratings=data,feedback=self.feedback,
                                  bias=bias, shape=self.shape, K=self.K,
                                  steps=self.steps, alpha=self.alpha,
                                  beta=self.beta, verbose=self.verbose,
                                  random_state=self.random_state)

        # Set as feedback the inferred feedback when no feedback has been given
        if self.feedback is not None:
            new_feedback = self.feedback

        # Construct model
        model = SVDPlusPlusModel(P=P, Q=Q, Y=Y, bias=bias,
                                 feedback=new_feedback)
        return super().prepare_model(model)


class SVDPlusPlusModel(Model):

    def __init__(self, P, Q, Y, bias, feedback):
        self.P = P
        self.Q = Q
        self.Y = Y
        self.bias = bias
        self.feedback = feedback
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
        idxs_missing = super().prepare_predict(X)

        users = X[:, self.order[0]]
        items = X[:, self.order[1]]

        predictions = np.zeros(len(X))
        feedback_cached = defaultdict(list)
        isFeedbackADict = isinstance(self.feedback, dict)

        for i in range(0, len(users)):
            u = users[i]

            if isFeedbackADict:
                feedback_u = self.feedback[u]
            else:
                feedback_u = save_in_cache(self.feedback, u, feedback_cached)
                feedback_u = feedback_u[feedback_u < self.shape[1]]  # For CV

            predictions[i] = _predict(u, items[i], self.bias['globalAvg'],
                              self.bias['dUsers'], self.bias['dItems'], self.P,
                              self.Q, self.Y, feedback_u)[0]

        # Set predictions for non-existing indices (CV)
        predictions = self.fix_predictions(predictions, self.bias, idxs_missing)
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
        # TODO: Cast rows, cols through preprocess
        objective = 0

        # Transform rating matrix into CSR sparse matrix
        data = sparse_matrix_2d(row=data.X[:, self.order[0]],
                                col=data.X[:, self.order[1]],
                                data=data.Y, shape=self.shape)

        # Compute predictions
        for u, j in zip(*data.nonzero()):
            feedback_u = self.feedback[u]

            # Prediction
            b_ui = self.bias['globalAvg'] + \
                   bias['dItems'][j] + \
                   bias['dUsers'][u]

            norm_denominator = math.sqrt(len(feedback_u))
            tempN = np.sum(Y[feedback_u], axis=0)
            p_plus_y_sum_vector = tempN / norm_denominator + P[u, :]

            rij_pred = b_ui + np.dot(p_plus_y_sum_vector, Q[j, :])

            objective += (rij_pred - data[u, j]) ** 2

            # Regularization
            objective += beta * (np.linalg.norm(P[u, :]) ** 2
                                 + np.linalg.norm(Q[j, :]) ** 2
                                 + bias['dItems'][j] ** 2
                                 + bias['dUsers'][u] ** 2)

        # Compute RMSE
        rmse = math.sqrt(objective / float(data.nnz))
        return rmse

    def getPTable(self):
        variable = self.original_domain.variables[self.order[0]]
        return feature_matrix(variable, self.P)

    def getQTable(self):
        variable = self.original_domain.variables[self.order[1]]
        return feature_matrix(variable, self.Q)

    def getYTable(self):
        domain_name = 'Feedback-feature'
        variable = self.original_domain.variables[self.order[0]]
        return feature_matrix(variable, self.Y, domain_name)



if __name__ == "__main__":
    import Orange

    print('Loading data...')
    ratings = Orange.data.Table('filmtrust/ratings.tab')

    start = time.time()
    learner = SVDPlusPlusLearner(K=15, steps=1, alpha=0.007, beta=0.1,
                                 verbose=True)
    recommender = learner(ratings)
    print('- Time (SVDPlusPlusLearner): %.3fs' % (time.time() - start))