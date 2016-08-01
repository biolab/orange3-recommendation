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


def _predict(u, j, global_avg, dUsers, dItems, P, Q, Y, W, feedback,
             trustees):
    bias = global_avg + dUsers[u] + dItems[j]

    # Implicit feedback
    norm_feedback = math.sqrt(len(feedback))
    if norm_feedback > 0:
        y_sum = np.sum(Y[feedback, :], axis=0)
        y_term = y_sum / norm_feedback
    else:
        y_term = 0

    # Trust information
    norm_trust = math.sqrt(len(trustees))
    if norm_trust > 0:
        w_sum = np.sum(W[trustees, :], axis=0)
        w_term = w_sum / norm_trust
    else:
        w_term = 0

    # Compute base
    p_enhanced = P[u, :] + (y_term + w_term)
    base_pred = np.einsum('i,i', p_enhanced, Q[j, :])

    return bias + base_pred, y_term, w_term, norm_feedback, norm_trust


def _predict_all_items(u, global_avg, dUsers, dItems, P, Q, Y, W, feedback,
                       trustees):
    bias = global_avg + dUsers[u] + dItems

    # Implicit feedback
    norm_feedback = math.sqrt(len(feedback))
    if norm_feedback > 0:
        y_sum = np.sum(Y[feedback, :], axis=0)
        y_term = y_sum / norm_feedback
    else:
        y_term = 0

    # Trust information
    norm_trust = math.sqrt(len(trustees))
    if norm_trust > 0:
        w_sum = np.sum(W[trustees, :], axis=0)
        w_term = w_sum / norm_trust
    else:
        w_term = 0

    # Compute base
    p_enhanced = P[u, :] + (y_term + w_term)

    base_pred = np.dot(p_enhanced, Q.T)
    return bias + base_pred


# TODO: Change name trust_users
def _matrix_factorization(ratings, feedback, trust, bias, shape, shape_t, K,
                          steps, alpha, beta, beta_t, verbose=False,
                          random_state=None):

    # Seed the generator
    if random_state is not None:
        np.random.seed(random_state)

    # Get featured matrices dimensions
    num_users, num_items = shape
    num_users = max(num_users, shape_t[0])

    # Initialize low-rank matrices
    P = np.random.rand(num_users, K)  # User-feature matrix
    Q = np.random.rand(num_items, K)  # Item-feature matrix
    Y = np.random.randn(num_items, K)  # Feedback-feature matrix
    W = np.random.randn(num_users, K)  # Trust-feature matrix

    # Compute bias (not need it if learnt)
    globalAvg = bias['globalAvg']
    dItems = bias['dItems']
    dUsers = bias['dUsers']

    # Cache rows
    # >>> From 2 days to 30s
    users_cached = defaultdict(list)
    trusters_cached = defaultdict(list)
    feedback_cached = defaultdict(list)

    # Cache norms (slower than list, but allows vectorization)
    # >>>  Lists: 6s; Arrays: 12s -> vectorized: 2s
    norm_I = np.zeros(num_items)
    norm_Tr = np.zeros(num_users)
    norm_Tc = np.zeros(num_users)

    # Precompute transpose (most costly operation)
    ratings_T = ratings.T
    trust_T = trust.T

    # Factorize matrix using SGD
    for step in range(steps):
        if verbose:
            start = time.time()
            print('- Step: %d' % (step + 1))
        
        # To update the gradients at the same time
        tempP = np.zeros(P.shape)
        tempQ = np.zeros(Q.shape)
        tempY = np.zeros(Y.shape)
        tempW = np.zeros(W.shape)

        # Optimize rating prediction
        objective = 0
        for u, j in zip(*ratings.nonzero()):

            # Store lists in cache
            items_rated_by_u = cache_rows(ratings, u, users_cached)
            trustees_u = cache_rows(trust, u, trusters_cached)

            # if there is no feedback, infer it from the ratings
            if feedback is None:
                feedback_u = items_rated_by_u
            else:
                feedback_u = cache_rows(feedback, u, feedback_cached)

            # Prediction and error
            ruj_pred, y_term, w_term, norm_feedback, norm_trust\
                = _predict(u, j, globalAvg, dUsers, dItems, P, Q, Y, W,
                           feedback_u, trustees_u)
            euj = ruj_pred - ratings[u, j]

            # Gradient P
            tempP[u, :] = euj * Q[j, :] + \
                          (beta/norm_feedback) * P[u, :]  # P: Part 1

            # Gradient Q
            norm_Uj = cache_norms(ratings_T, j, norm_I)
            tempQ[j, :] = euj * (P[u, :] + y_term + w_term) + \
                          (beta/norm_Uj) * Q[j, :]

            # Gradient Y
            if norm_feedback > 0:
                tempY1 = (euj/norm_feedback) * Q[j, :]
                norms = cache_norms(ratings_T, items_rated_by_u, norm_I)
                norm_b = (beta/np.atleast_2d(norms))
                tempY[items_rated_by_u, :] = tempY1 + \
                                      np.multiply(norm_b.T, Y[items_rated_by_u, :])

            # Gradient W
            if norm_trust > 0:
                tempW1 = (euj/norm_trust) * Q[j, :]  # W: Part 1
                norms = cache_norms(trust_T, trustees_u, norm_Tc)
                norm_b = (beta/np.atleast_2d(norms))
                tempW[trustees_u, :] = tempW1 + \
                                      np.multiply(norm_b.T, W[trustees_u, :])

        # Optimize trust prediction
        for u, v in zip(*trust.nonzero()):

            tuv_pred = np.dot(W[v, :], P[u, :])
            euv = tuv_pred - trust[u, v]

            # Gradients of P and W
            norm_trust = cache_norms(trust, u, norm_Tr)

            tempP[u, :] += beta_t * \
                           (euv * W[v, :] + P[u, :]/norm_trust)  # P: Part 2
            tempW[v, :] += beta_t * euv * P[u, :]  # W: Part 2

        P -= alpha * tempP
        Q -= alpha * tempQ
        Y -= alpha * tempY
        W -= alpha * tempW

        # TODO: Loss function

    if verbose:
        print('\t- Loss: %.3f' % objective)
        print('\t- Time: %.3fs' % (time.time() - start))
        print('')

    if feedback is None:
        feedback = users_cached

    return P, Q, Y, W, feedback


class TrustSVDLearner(Learner):
    """Trust-based matrix factorization

    This model uses stochastic gradient descent to find four low-rank
    matrices: user-feature matrix, item-feature matrix, feedback-feature matrix
    and trustee-feature matrix.

    Attributes:
        K: int, optional
            The number of latent factors.

        steps: int, optional
            The number of passes over the training data (aka epochs).

        alpha: float, optional
            The learning rate.

        beta: float, optional
            The regularization for the ratings.

        beta_t: float, optional
            The regularization for the trust.

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

    def __init__(self, K=5, steps=25, alpha=0.07, beta=0.1, beta_t=0.05,
                 min_rating=None, max_rating=None, feedback=None, trust=None,
                 preprocessors=None, verbose=False, random_state=None):
        self.K = K
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.beta_t = beta_t
        self.random_state = random_state

        self.feedback = feedback
        if feedback is not None:
            self.feedback, order_f, self.shape_f = preprocess(feedback)

            # Transform feedback matrix into a sparse matrix
            self.feedback = table2sparse(self.feedback, self.shape_f,
                                         order_f, type=__sparse_format__)

        self.trust = trust
        if trust is not None:
            self.trust, order_t, self.shape_t = preprocess(trust)
            max_trow = int(np.max(self.trust.X[:, order_t[0]]))
            max_tcol = int(np.max(self.trust.X[:, order_t[1]]))
            self.shape_t = (max_trow + 1, max_tcol + 1)

            # Transform trust matrix into a sparse matrix
            self.trust = table2sparse(self.trust, self.shape_t, order_t,
                                      type=__sparse_format__)

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
        P, Q, Y, W, new_feedback = \
            _matrix_factorization(ratings=data, feedback=self.feedback,
                                  trust=self.trust, bias=bias,
                                  shape=self.shape, shape_t=self.shape_t,
                                  K=self.K, steps=self.steps, alpha=self.alpha,
                                  beta=self.beta, beta_t=self.beta_t,
                                  verbose=self.verbose,
                                  random_state=self.random_state)

        # Set as feedback the inferred feedback when no feedback has been given
        if self.feedback is not None:
            new_feedback = self.feedback

        # Construct model
        model = TrustSVDModel(P=P, Q=Q, Y=Y, W=W, bias=bias,
                              feedback=new_feedback, trust=self.trust)
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
        idxs_missing = super().prepare_predict(X)

        users = X[:, self.order[0]]
        items = X[:, self.order[1]]

        predictions = np.zeros(len(X))

        trusters_cached = defaultdict(list)
        feedback_cached = defaultdict(list)
        isFeedbackADict = isinstance(self.feedback, dict)

        for i in range(0, len(users)):
            u = users[i]

            trustees_u = cache_rows(self.trust, u, trusters_cached)

            if isFeedbackADict:
                feedback_u = self.feedback[u]
            else:
                feedback_u = cache_rows(self.feedback, u, feedback_cached)

            predictions[i] = _predict(u, items[i], self.bias['globalAvg'],
                              self.bias['dUsers'], self.bias['dItems'], self.P,
                              self.Q, self.Y, self.W, feedback_u, trustees_u)[0]

        # Set predictions for non-existing indices (CV)
        predictions = self.fix_predictions(predictions, self.bias, idxs_missing)
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
        trusters_cached = defaultdict(list)
        feedback_cached = defaultdict(list)
        isFeedbackADict = isinstance(self.feedback, dict)

        for i in range(0, len(users)):
            u = users[i]

            trustees_u = cache_rows(self.trust, u, trusters_cached)

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



if __name__ == "__main__":
    import Orange
    from sklearn.metrics import mean_squared_error

    print('Loading data...')
    ratings = Orange.data.Table('filmtrust/ratings.tab')
    trust = Orange.data.Table('filmtrust/trust.tab')

    start = time.time()
    learner = TrustSVDLearner(K=15, steps=1, alpha=0.07, beta=0.1, beta_t=0.05,
                              trust=trust, verbose=True)
    recommender = learner(ratings)
    print('- Time (TrustSVD): %.3fs' % (time.time() - start))

    # prediction = recommender.predict(ratings, trust)
    # rmse = math.sqrt(mean_squared_error(ratings.Y, ))
    # print('- RMSE (SVDPlusPlusLearner): %.3f' % rmse)
