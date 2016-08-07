from orangecontrib.recommendation.rating import Learner, Model
from orangecontrib.recommendation.utils.format_data import *
from orangecontrib.recommendation.utils.datacaching \
    import cache_norms, cache_rows

from collections import defaultdict

import numpy as np
import math
import time
import warnings

__all__ = ['TrustSVDLearner']
__sparse_format__ = lil_matrix


def _compute_extra_terms(Y, W, items_u, trustees_u):
    # Implicit information
    norm_Iu = math.sqrt(len(items_u))
    if norm_Iu > 0:
        y_sum = np.sum(Y[items_u, :], axis=0)
        y_term = y_sum / norm_Iu
    else:
        y_term = 0

    # Trust information
    norm_Tu = math.sqrt(len(trustees_u))
    if norm_Tu > 0:
        w_sum = np.sum(W[trustees_u, :], axis=0)
        w_term = w_sum / norm_Tu
    else:
        w_term = 0

    return y_term, w_term, norm_Iu, norm_Tu


def _predict(u, j, global_avg, bu, bi, P, Q, Y, W, items_u, trustees_u):
    # Compute bias
    bias = global_avg + bu[u] + bi[j]

    # Compute extra terms
    y_term, w_term, norm_Iu, norm_Tu = \
        _compute_extra_terms(Y, W, items_u, trustees_u)

    # Compute base
    p_enhanced = P[u, :] + (y_term + w_term)
    base_pred = np.einsum('i,i', p_enhanced, Q[j, :])

    # Compute prediction and return extra terms and norms
    return bias + base_pred, y_term, w_term, norm_Iu, norm_Tu


def _predict_all_items(u, global_avg, bu, bi, P, Q, Y, W, items_u, trustees_u):
    # Compute bias
    bias = global_avg + bu[u] + bi

    # Compute extra terms
    y_term, w_term, _, _ = _compute_extra_terms(Y, W, items_u, trustees_u)

    # Compute base
    p_enhanced = P[u, :] + (y_term + w_term)

    # Compute prediction
    base_pred = np.dot(p_enhanced, Q.T)
    return bias + base_pred


def _matrix_factorization(ratings, trust, bias, shape, shape_t, num_factors, 
                          num_iter, learning_rate, bias_learning_rate, lmbda, 
                          bias_lmbda, social_lmbda, verbose=False, 
                          random_state=None):

    # Seed the generator
    if random_state is not None:
        np.random.seed(random_state)

    # Get featured matrices dimensions
    num_users, num_items = shape
    num_users = max(num_users, max(shape_t))

    # Initialize low-rank matrices
    P = np.random.rand(num_users, num_factors)  # User-feature matrix
    Q = np.random.rand(num_items, num_factors)  # Item-feature matrix
    Y = np.random.randn(num_items, num_factors)  # Feedback-feature matrix
    W = np.random.randn(num_users, num_factors)  # Trust-feature matrix

    # Compute bias (not need it if learnt)
    global_avg = bias['globalAvg']
    bu = bias['dUsers']
    bi = bias['dItems']

    # Cache rows
    # >>> From 2 days to 30s
    users_cache = defaultdict(list)
    trusters_cache = defaultdict(list)

    # Cache norms (slower than list, but allows vectorization)
    # >>>  Lists: 6s; Arrays: 12s -> vectorized: 2s
    norm_I = np.zeros(num_users)  # norms of Iu
    norm_Tr = np.zeros(num_users)  # norms of Tu
    norm_Tc = np.zeros(num_users)  # norms of Tv

    # Precompute transpose (most costly operation)
    ratings_T = ratings.T
    trust_T = trust.T

    # Factorize matrix using SGD
    for step in range(num_iter):
        if verbose:
            start = time.time()
            print('- Step: %d' % (step + 1))

        # To update the gradients at the same time
        tempP = np.zeros(P.shape)
        tempQ = np.zeros(Q.shape)
        tempY = np.zeros(Y.shape)
        tempW = np.zeros(W.shape)

        # Optimize rating prediction
        for u, j in zip(*ratings.nonzero()):

            # Store lists in cache
            items_u = cache_rows(ratings, u, users_cache)
            trustees_u = cache_rows(trust, u, trusters_cache)
            # No need to cast for CV because of "max(num_users, shape_t[0])"

            # Prediction and error
            ruj_pred, y_term, w_term, norm_Iu, norm_Tu = \
                _predict(u, j, global_avg, bu, bi, P, Q, Y, W, items_u,
                           trustees_u)
            euj = ruj_pred - ratings[u, j]

            # Store/Compute norms
            norm_I[u] = norm_Iu
            norm_Tr[u] = norm_Tu
            norm_Uj = cache_norms(ratings_T, j, norm_I)

            # Gradient Bu
            reg_bu = (bias_lmbda/norm_Iu) * bu[u] if norm_Iu > 0 else 0
            bu[u] -= bias_learning_rate * (euj + reg_bu)

            # Gradient Bi
            reg_bi = (bias_lmbda/norm_Uj) * bi[j] if norm_Uj > 0 else 0
            bi[j] -= bias_learning_rate * (euj + reg_bi)

            # Gradient P
            reg_p = (lmbda/norm_Iu) * P[u, :] if norm_Iu > 0 else 0
            tempP[u, :] = euj * Q[j, :] + reg_p  # P: Part 1

            # Gradient Q
            reg_q = (lmbda/norm_Uj) * Q[j, :] if norm_Uj > 0 else 0
            tempQ[j, :] = euj * (P[u, :] + y_term + w_term) + reg_q

            # Gradient Y
            if norm_Iu > 0:
                tempY1 = (euj/norm_Iu) * Q[j, :]
                norms = cache_norms(ratings_T, items_u, norm_I)
                norm_b = (lmbda/np.atleast_2d(norms))
                tempY[items_u, :] = tempY1 + \
                                np.multiply(norm_b.T, Y[items_u, :])

            # Gradient W
            if norm_Tu > 0:
                tempW1 = (euj/norm_Tu) * Q[j, :]  # W: Part 1
                norms = cache_norms(trust_T, trustees_u, norm_Tc)
                norm_b = (lmbda/np.atleast_2d(norms))
                tempW[trustees_u, :] = tempW1 + \
                                      np.multiply(norm_b.T, W[trustees_u, :])

        # Optimize trust prediction
        for u, v in zip(*trust.nonzero()):

            # Prediction and error
            tuv_pred = np.dot(W[v, :], P[u, :])
            euv = tuv_pred - trust[u, v]

            # Gradient P (Part 2)
            norm_Tu = cache_norms(trust, u, norm_Tr)
            reg_p = P[u, :]/norm_Tu if norm_Tu > 0 else 0
            tempP[u, :] += social_lmbda * (euv * W[v, :] + reg_p)

            # Gradient W (Part 2)
            tempW[v, :] += social_lmbda * euv * P[u, :]  # W: Part 2

        P -= learning_rate * tempP
        Q -= learning_rate * tempQ
        Y -= learning_rate * tempY
        W -= learning_rate * tempW

        if verbose:
            # Set parameters and compute loss
            data_t = (ratings, trust)
            bias_t = (global_avg, bu, bi)
            low_rank_matrices = (P, Q, Y, W)
            params = (lmbda, bias_lmbda, social_lmbda)
            objective = compute_loss(data_t, bias_t, low_rank_matrices, params)

            print('\t- Loss: %.3f' % objective)
            print('\t- Time: %.3fs' % (time.time() - start))
            print('')

    return P, Q, Y, W, bu, bi, users_cache


def compute_loss(data, bias, low_rank_matrices, params):

    # Set parameters
    ratings, trust = data
    global_avg, bu, bi = bias
    P, Q, Y, W = low_rank_matrices
    lmbda, bias_lmbda, social_lmbda = params

    # Check data type
    if isinstance(ratings, __sparse_format__):
        pass
    elif isinstance(ratings, Table):
        # Preprocess Orange.data.Table and transform it to sparse
        ratings, order, shape = preprocess(ratings)
        ratings = table2sparse(ratings, shape, order, m_type=__sparse_format__)
    else:
        raise TypeError('Invalid data type')

    # Check data type
    if isinstance(trust, dict) or isinstance(trust, __sparse_format__):
        pass
    elif isinstance(trust, Table):
        # Preprocess Orange.data.Table and transform it to sparse
        trust, order, shape = preprocess(trust)
        trust = table2sparse(trust, shape, order, m_type=__sparse_format__)
    else:
        raise TypeError('Invalid data type')

    # Get featured matrices dimensions
    num_users, num_items = ratings.shape
    num_users = max(num_users, max(trust.shape))

    # Cache rows
    # >>> From 2 days to 30s
    users_cache = defaultdict(list)
    trusters_cache = defaultdict(list)

    # Cache norms (slower than list, but allows vectorization)
    # >>>  Lists: 6s; Arrays: 12s -> vectorized: 2s
    norm_I = np.zeros(num_users)  # norms of Iu
    norm_U = np.zeros(num_items)  # norms of Uj
    norm_Tr = np.zeros(num_users)  # norms of Tu
    norm_Tc = np.zeros(num_users)  # norms of Tv

    # Precompute transpose (most costly operation)
    ratings_T = ratings.T
    trust_T = trust.T

    # Optimize rating prediction
    objective = 0
    for u, j in zip(*ratings.nonzero()):

        # Store lists in cache
        items_u = cache_rows(ratings, u, users_cache)
        trustees_u = cache_rows(trust, u, trusters_cache)

        # Prediction and error
        ruj_pred, _, _, norm_Iu, norm_Tu = \
            _predict(u, j, global_avg, bu, bi, P, Q, Y, W, items_u, trustees_u)

        # Cache norms
        norm_I[u] = norm_Iu
        norm_Tr[u] = norm_Tu

        # Compute loss
        objective += 0.5 * (ruj_pred - ratings[u, j])**2

    # Optimize trust prediction
    for u, v in zip(*trust.nonzero()):
        # Prediction
        tuv_pred = np.dot(W[v, :], P[u, :])

        # Compute loss
        objective += social_lmbda * 0.5 * (tuv_pred - trust[u, v])**2

    for u in range(P.shape[0]):   # users
        # Cache norms
        norm_Iu = cache_norms(ratings, u, norm_I)
        norm_Tu = cache_norms(trust, u, norm_Tr)
        norm_Tv = cache_norms(trust_T, u, norm_Tc)

        # Compute loss
        term_l = 0
        if norm_Iu > 0:
            objective += bias_lmbda/(2*norm_Iu) * bu[u]**2
            term_l = lmbda/(2*norm_Iu)

        term_s = 0
        if norm_Tu > 0:
            term_s = social_lmbda/(2 * norm_Tu)

        term_ls = term_l + term_s
        if term_ls > 0:
            objective += term_ls * np.linalg.norm(P[u, :])**2

        if norm_Tv > 0:
            objective += lmbda/(2*norm_Tv) * np.linalg.norm(W[u, :])**2

    for j in range(Q.shape[0]):   # items
        # Cache norms
        norm_Uj = cache_norms(ratings_T, j, norm_U)

        # Compute loss
        if norm_Uj > 0:
            objective += bias_lmbda/(2*norm_Uj) * bi[j]**2
            objective += lmbda/(2*norm_Uj) * np.linalg.norm(Q[j, :])**2
            objective += lmbda/(2*norm_Uj) * np.linalg.norm(Y[j, :])**2

    return objective


class TrustSVDLearner(Learner):
    """Trust-based matrix factorization

    This model uses stochastic gradient descent to find four low-rank
    matrices: user-feature matrix, item-feature matrix, feedback-feature matrix
    and trustee-feature matrix.

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

        social_lmbda: float, optional
            Controls the importance of the trust regularization term.

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

        trust: Orange.data.Table
            Social trust information.

        verbose: boolean, optional
            Prints information about the process.

        random_state: int, optional
            Set the seed for the numpy random generator, so it makes the random
            numbers predictable. This a debbuging feature.

    """

    name = 'TrustSVD'

    def __init__(self, num_factors=5, num_iter=25, learning_rate=0.07,
                 bias_learning_rate=None, lmbda=0.1, bias_lmbda=None,
                 social_lmbda=0.05, min_rating=None, max_rating=None,
                 trust=None, preprocessors=None, verbose=False,
                 random_state=None):
        self.num_factors = num_factors
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.bias_learning_rate = bias_learning_rate
        self.lmbda = lmbda
        self.bias_lmbda = bias_lmbda
        self.social_lmbda = social_lmbda
        self.random_state = random_state

        # Correct assignments
        if self.bias_learning_rate is None:
            self.bias_learning_rate = self.learning_rate
        if self.bias_lmbda is None:
            self.bias_lmbda = self.lmbda

        self.trust = trust
        if trust is not None:
            self.trust, order_t, self.shape_t = preprocess(trust)
            max_trow = int(np.max(self.trust.X[:, order_t[0]]))
            max_tcol = int(np.max(self.trust.X[:, order_t[1]]))
            self.shape_t = (max_trow + 1, max_tcol + 1)

            # Transform trust matrix into a sparse matrix
            self.trust = table2sparse(self.trust, self.shape_t, order_t,
                                      m_type=__sparse_format__)

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
                            m_type=__sparse_format__)
        # Factorize matrix
        P, Q, Y, W, bu, bi, temp_feedback = \
            _matrix_factorization(ratings=data, trust=self.trust, bias=bias,
                                  shape=self.shape, shape_t=self.shape_t,
                                  num_factors=self.num_factors,
                                  num_iter=self.num_iter,
                                  learning_rate=self.learning_rate,
                                  bias_learning_rate=self.bias_learning_rate,
                                  lmbda=self.lmbda,
                                  bias_lmbda=self.bias_lmbda,
                                  social_lmbda=self.social_lmbda,
                                  verbose=self.verbose,
                                  random_state=self.random_state)

        # Update biases
        bias['dUsers'] = bu
        bias['dItems'] = bi

        # Construct model
        model = TrustSVDModel(P=P, Q=Q, Y=Y, W=W, bias=bias,
                              feedback=temp_feedback, trust=self.trust)
        return super().prepare_model(model)


class TrustSVDModel(Model):

    def __init__(self, P, Q, Y, W, bias, feedback, trust):
        self.P = P
        self.Q = Q
        self.Y = Y
        self.W = W
        self.bias = bias
        self.feedback = feedback
        self.trust = trust
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

        trusters_cache = defaultdict(list)
        feedback_cached = defaultdict(list)
        isFeedbackADict = isinstance(self.feedback, dict)

        for i in range(0, len(users)):
            u = users[i]

            trustees_u = cache_rows(self.trust, u, trusters_cache)
            # No need to cast for CV because of "max(num_users, shape_t[0])"

            if isFeedbackADict:
                feedback_u = self.feedback[u]
            else:
                feedback_u = cache_rows(self.feedback, u, feedback_cached)
                feedback_u = feedback_u[feedback_u < self.shape[1]]  # For CV

            predictions[i] = _predict(u, items[i], self.bias['globalAvg'],
                              self.bias['dUsers'], self.bias['dItems'], self.P,
                              self.Q, self.Y, self.W, feedback_u, trustees_u)[0]

        # Set predictions for non-existing indices (CV)
        predictions = self.fix_predictions(X, predictions, self.bias)
        return super().predict_on_range(np.asarray(predictions))

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

        if users is None:
            users = np.asarray(range(0, len(self.bias['dUsers'])))

        predictions = []
        trusters_cache = defaultdict(list)
        feedback_cached = defaultdict(list)
        isFeedbackADict = isinstance(self.feedback, dict)

        for i in range(0, len(users)):
            u = users[i]

            trustees_u = cache_rows(self.trust, u, trusters_cache)

            if isFeedbackADict:
                feedback_u = self.feedback[u]
            else:
                feedback_u = cache_rows(self.feedback, u, feedback_cached)

            pred = _predict_all_items(u, self.bias['globalAvg'],
                                      self.bias['dUsers'], self.bias['dItems'],
                                      self.P, self.Q, self.Y, self.W,
                                      feedback_u, trustees_u)
            predictions.append(pred)

        predictions = np.asarray(predictions)

        # Return top-k recommendations
        if top is not None:
            predictions = predictions[:, :top]

        return super().predict_on_range(predictions)

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

    def getWTable(self):
        # TODO Correct variable type
        domain_name = 'Trust-feature'
        variable = self.original_domain.variables[self.order[0]]
        return feature_matrix(variable, self.W, domain_name)



# if __name__ == "__main__":
#     import Orange
#     from sklearn.metrics import mean_squared_error
#
#     print('Loading data...')
#     ratings = Orange.data.Table('filmtrust/ratings.tab')
#     trust = Orange.data.Table('filmtrust/trust.tab')
#
#     start = time.time()
#     learner = TrustSVDLearner(num_factors=15, num_iter=1, learning_rate=0.07,
#                               lmbda=0.1, social_lmbda=0.05,
#                               trust=trust, verbose=True)
#     recommender = learner(ratings)
#     print('- Time (TrustSVD): %.3fs' % (time.time() - start))
#
#     # prediction = recommender.predict(ratings, trust)
#     # rmse = math.sqrt(mean_squared_error(ratings.Y, ))
#     # print('- RMSE (SVDPlusPlusLearner): %.3f' % rmse)
