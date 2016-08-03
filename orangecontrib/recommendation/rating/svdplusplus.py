from orangecontrib.recommendation.rating import Learner, Model
from orangecontrib.recommendation.utils.format_data import *

from collections import defaultdict

import numpy as np
import math
import time
import warnings

__all__ = ['SVDPlusPlusLearner']
__sparse_format__ = lil_matrix


def _predict(u, j, global_avg, bu, bi, P, Q, Y, feedback):
    bias = global_avg + bu[u] + bi[j]

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


def _predict_all_items(u, global_avg, bu, bi, P, Q, Y, feedback):
    bias = global_avg + bu[u] + bi

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


def _matrix_factorization(ratings, feedback, bias, shape, num_factors, num_iter,
                          learning_rate, bias_learning_rate, lmbda, bias_lmbda,
                          verbose=False, random_state=None):

    # Seed the generator
    if random_state is not None:
        np.random.seed(random_state)

    # Get featured matrices dimensions
    num_users, num_items = shape

    # Initialize low-rank matrices
    P = np.random.rand(num_users, num_factors)  # User-feature matrix
    Q = np.random.rand(num_items, num_factors)  # Item-feature matrix
    Y = np.random.randn(num_items, num_factors) # Feedback-feature matrix

    # Compute bias (not need it if learnt)
    global_avg = bias['globalAvg']
    bu = bias['dUsers']
    bi = bias['dItems']

    # Cache rows
    users_cached = defaultdict(list)
    feedback_cached = defaultdict(list)

    # Factorize matrix using SGD
    for step in range(num_iter):
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
                _predict(u, j, global_avg, bu, bi, P, Q, Y, feedback_u)
            eij = ruj_pred - ratings[u, j]

            # Compute gradients
            tempBu = eij + bias_lmbda * bu[u]
            tempBi = eij + bias_lmbda * bi[j]
            tempP = eij * Q[j, :] - lmbda * P[u, :]
            tempQ = eij * (P[u, :] + y_term) - lmbda * Q[j, :]

            # Gradient Y
            if norm_feedback > 0:
                for i in feedback_u:
                    Y[i, :] -= bias_learning_rate * (eij/norm_feedback * Q[j, :]
                                                     - lmbda * Y[i, :])

            # Update the gradients at the same time
            # I use the loss function divided by 2, to simplify the gradients
            bu[u] -= bias_learning_rate * tempBu
            bi[j] -= bias_learning_rate * tempBi
            P[u, :] -= learning_rate * tempP
            Q[j, :] -= learning_rate * tempQ

            # Loss function (Remember it must be divided by 2 to be correct)
            if verbose:
                objective += eij ** 2
                temp_y = np.sum(Y[feedback_u, :], axis=0)
                objective += lmbda * (np.linalg.norm(P[u, :]) ** 2 +
                                      np.linalg.norm(Q[j, :]) ** 2 +
                                      np.linalg.norm(temp_y) ** 2) + \
                             bias_lmbda * (bu[u] ** 2 + bi[j] ** 2)

        if verbose:
            print('\t- Loss: %.3f' % (objective*0.5))
            print('\t- Time: %.3fs' % (time.time() - start))
            print('')

    if feedback is None:
        feedback = users_cached

    return P, Q, Y, bu, bi, feedback


class SVDPlusPlusLearner(Learner):
    """SVD++ matrix factorization

    This model uses stochastic gradient descent to find three low-rank
    matrices: user-feature matrix, item-feature matrix and feedback-feature
    matrix.

    Attributes:
        num_factors: int, optional
            The number of latent factors.

        num_iter: int, optional
            The number of passes over the training data (aka epochs).

        learning_rate: float, optional
            The learning rate controlling the size of update steps (general).

        bias_learning_rate: float, optional
            The learning rate controlling the size of the bias update steps.
            If None (default), bias_learning_rate = learning_rate

        lmbda: float, optional
            Controls the importance of the regularization term (general).
            Avoids overfitting by penalizing the magnitudes of the parameters.

        bias_lmbda: float, optional
            Controls the importance of the bias regularization term.
            If None (default), bias_lmbda = lmbda

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

    def __init__(self, num_factors=5, num_iter=25, learning_rate=0.07,
                 bias_learning_rate=None, lmbda=0.1, bias_lmbda=None,
                 min_rating=None, max_rating=None, feedback=None,
                 preprocessors=None, verbose=False, random_state=None):
        self.num_factors = num_factors
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.bias_learning_rate = bias_learning_rate
        self.lmbda = lmbda
        self.bias_lmbda = bias_lmbda
        self.random_state = random_state

        # Correct assignments
        if self.bias_learning_rate is None:
            self.bias_learning_rate = self.learning_rate
        if self.bias_lmbda is None:
            self.bias_lmbda = self.lmbda

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
        if self.learning_rate == 0:
            warnings.warn("With learning_rate=0, this algorithm does not "
                          "converge well.", stacklevel=2)

        # Compute biases (not need it if learnt)
        bias = self.compute_bias(data, 'all')

        # Transform ratings matrix into a sparse matrix
        data = table2sparse(data, self.shape, self.order,
                            type=__sparse_format__)

        # Factorize matrix
        P, Q, Y, bu, bi, temp_feedback = \
            _matrix_factorization(ratings=data, feedback=self.feedback,
                                  bias=bias, shape=self.shape,
                                  num_factors=self.num_factors,
                                  num_iter=self.num_iter,
                                  learning_rate=self.learning_rate,
                                  bias_learning_rate=self.bias_learning_rate,
                                  lmbda=self.lmbda,
                                  bias_lmbda=self.bias_lmbda,
                                  verbose=self.verbose,
                                  random_state=self.random_state)

        # Update biases
        bias['dUsers'] = bu
        bias['dItems'] = bi

        # Set as feedback the inferred feedback when no feedback has been given
        if self.feedback is not None:
            temp_feedback = self.feedback

        # Construct model
        model = SVDPlusPlusModel(P=P, Q=Q, Y=Y, bias=bias,
                                 feedback=temp_feedback)
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
        super().prepare_predict(X)

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

    def compute_objective(self, data, lmbda, bias_lmbda):
        # TODO: Cast rows, cols through preprocess
        objective = 0

        # Transform rating matrix into CSR sparse matrix
        data = sparse_matrix_2d(row=data.X[:, self.order[0]],
                                col=data.X[:, self.order[1]],
                                data=data.Y, shape=self.shape)

        global_avg = self.bias['globalAvg']
        bi = self.bias['dItems']
        bu = self.bias['dUsers']

        # Compute predictions
        for u, j in zip(*data.nonzero()):
            # Prediction
            ruj_pred, _, _ = \
                _predict(u, j, global_avg, bu, bi, self.P, self.Q, self.Y,
                         self.feedback[u])
            objective += (ruj_pred - data[u, j]) ** 2

            # Regularization
            temp_y = np.sum(self.Y[self.feedback[u], :], axis=0)
            objective += lmbda * (np.linalg.norm(self.P[u, :]) ** 2 +
                                  np.linalg.norm(self.Q[j, :]) ** 2 +
                                  np.linalg.norm(temp_y) ** 2) + \
                         bias_lmbda * (bu[u] ** 2 + bi[j] ** 2)

        # Compute RMSE
        rmse = math.sqrt(objective*0.5 / float(data.nnz))
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
    learner = SVDPlusPlusLearner(num_factors=15, num_iter=1,
                                 learning_rate=0.007, lmbda=0.1, verbose=True)
    recommender = learner(ratings)
    print('- Time (SVDPlusPlusLearner): %.3fs' % (time.time() - start))