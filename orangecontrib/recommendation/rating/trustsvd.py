from orangecontrib.recommendation.rating import Learner, Model
from orangecontrib.recommendation.utils import format_data

import numpy as np
from scipy import sparse
from collections import defaultdict
import math
import sys
import time
import warnings

__all__ = ['TrustSVDLearner']


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


def save_in_cache(matrix, key, cache):
    if key not in cache:
        if key < matrix.shape[0]:
            cache[key] = matrix.rows[key]
        else:
            cache[key] = []
    return cache[key]


def _matrix_factorization(ratings, feedback, trust, bias, shape, trust_users,
                          order, K, steps, alpha, beta, beta_trust,
                          verbose=False, random_state=None):
    """ Factorize either a dense matrix or a sparse matrix into two low-rank
        matrices which represents user and item factors.

       Args:
           data: Sparse

           K: int
               The number of latent factors.

           steps: int
               The number of epochs of stochastic gradient descent.

           alpha: float
               The learning rate of stochastic gradient descent.

           beta: float
               The regularization parameter (general purpose).

           beta_trust: float
               The regularization parameter for the trust.

           random_state:
               Random state or None.

           verbose: boolean, optional
               If true, it outputs information about the process.

       Returns:
           P (matrix, UxK), Q (matrix, KxI) and bias (dictionary, 'delta items'
           , 'delta users')

       """

    if random_state is not None:
        np.random.seed(random_state)

    # Initialize factorized matrices randomly
    num_users, num_items = shape
    num_users = max(num_users, trust_users)

    P = np.random.rand(num_users, K)  # User and features
    Q = np.random.rand(num_items, K)  # Item and features
    Y = np.random.randn(num_items, K)
    W = np.random.randn(num_users, K)

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

        tempP = np.zeros(P.shape)
        tempQ = np.zeros(Q.shape)
        tempY = np.zeros(Y.shape)
        tempW = np.zeros(W.shape)

        users_cached = defaultdict(list)
        items_cached = defaultdict(list)
        trusters_cached = defaultdict(list)
        trustees_cached = defaultdict(list)
        feedback_cached = defaultdict(list)

        if verbose:
            start2 = time.time()

        # Optimize rating prediction
        for u, j in zip(*ratings.nonzero()):
            if verbose:
                start2 = time.time()

            items_rated_by_u = save_in_cache(ratings, u, users_cached)
            users_who_rated_j = save_in_cache(ratings.T, j, items_cached)
            trustees_u = save_in_cache(trust, u, trusters_cached)

            # if there is no feedback, infer it from the ratings
            if feedback is None:
                feedback_u = items_rated_by_u
            else:
                feedback_u = save_in_cache(feedback, u, feedback_cached)

            # Prediction
            ruj_pred, y_term, w_term, norm_feedback, norm_trust\
                = _predict(u, j, globalAvg, dUsers, dItems, P, Q, Y, W,
                           feedback_u, trustees_u)
            euj = ruj_pred - ratings[u, j]

            # Gradient P
            tempP[u, :] = euj * Q[j, :] + \
                          beta * (1/norm_feedback) * P[u, :]  # P: Part 1

            # Gradient Q
            tempQ[j, :] = euj * (P[u, :] + y_term + w_term) + \
                          beta * Q[j, :]/math.sqrt(len(users_who_rated_j))

            # Gradient Y
            if norm_feedback > 0:
                tempY1 = euj * (1/norm_feedback) * Q[j, :]
                for i in items_rated_by_u:
                    users_who_rated_i = save_in_cache(ratings.T, i, items_cached)
                    norm_Ui = math.sqrt(len(users_who_rated_i))
                    tempY[i, :] = tempY1 + (beta/norm_Ui) * Y[i, :]

            # Gradient W
            if norm_trust > 0:
                tempW1 = euj * (1/norm_trust) * Q[j, :]  # W: Part 1
                for v in trustees_u:
                    trusters_v = save_in_cache(trust.T, v, trustees_cached)
                    norm_Tv = math.sqrt(len(trusters_v))
                    tempW[v, :] = tempW1 + (beta/norm_Tv) * W[v, :]
            if verbose:
                print('\tTime iter rating: %.3fs' % (time.time() - start2))
        # sys.exit(0)

        # Optimize trust prediction
        for u, v in zip(*trust.nonzero()):
            if verbose:
                start2 = time.time()

            tuv_pred = np.dot(W[v, :], P[u, :])
            euv = tuv_pred - trust[u, v]

            # Get indices of the users who are trusted by u
            trustees_u = save_in_cache(trust, u, trusters_cached)
            norm_trust = math.sqrt(len(trustees_u))

            # Gradient of P and W
            tempP[u, :] += beta_trust * euv * W[v, :] + \
                           beta_trust * (1/norm_trust) * P[u, :]  # P: Part 2
            tempW[v, :] += beta_trust * euv * P[u, :]  # W: Part 2

            if verbose:
                print('\tTime iter trust: %.3fs' % (time.time() - start2))

        P -= alpha * tempP
        Q -= alpha * tempQ
        Y -= alpha * tempY
        W -= alpha * tempW

        if verbose:
            print('\tTime step: %.3fs' % (time.time() - start))
            #sys.exit(0)

    if feedback is None:
        feedback = users_cached

    return P, Q, Y, W, feedback


class TrustSVDLearner(Learner):
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

    name = 'TrustSVD'

    def __init__(self, K=5, steps=25, alpha=0.07, beta=0.1, beta_trust=0.05,
                 min_rating=None, max_rating=None, feedback=None, trust=None,
                 preprocessors=None, verbose=False, random_state=None):
        self.K = K
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.beta_trust = beta_trust
        self.feedback = feedback
        self.trust = trust
        self.trust_users = None
        self.random_state = random_state
        self.P = None
        self.Q = None
        self.Y = None
        self.W = None
        self.bias = None
        super().__init__(preprocessors=preprocessors, verbose=verbose,
                         min_rating=min_rating, max_rating=max_rating)

    def fit_storage(self, data):
        """This function calls the factorization method.

        Args:
            data: Orange.data.Table

        Returns:
            Model object (TrustSVDModel).

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

        # Transform trust matrix to sparse
        max_row = int(np.max(self.trust.X[:, 0]))
        max_col = int(np.max(self.trust.X[:, 1]))
        self.trust_users = int(max(max_row, max_col)) + 1
        self.trust = sparse.csr_matrix((self.trust.Y,
                                        (self.trust.X[:, 0],
                                         self.trust.X[:, 1])),
                                       shape=(max_row + 1,
                                              max_col + 1)).tolil()

        # Factorize matrix
        self.P, self.Q, self.Y, self.W, self.feedback = \
            _matrix_factorization(ratings=data, feedback=self.feedback,
                                  trust=self.trust, bias=self.bias,
                                  shape=self.shape, trust_users=self.trust_users,
                                  order=self.order, K=self.K,
                                  steps=self.steps, alpha=self.alpha,
                                  beta=self.beta, beta_trust=self.beta_trust,
                                  verbose=self.verbose,
                                  random_state=self.random_state)

        # Build model
        model = TrustSVDModel(P=self.P, Q=self.Q, Y=self.Y, W=self.W,
                              bias=self.bias, feedback=self.feedback,
                              trust=self.trust)
        return super().prepare_model(model)


class TrustSVDModel(Model):
    def __init__(self, P, Q, Y, W, bias, feedback, trust):
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
        self.Y = Y
        self.W = W
        self.bias = bias
        self.feedback = feedback
        self.trust = trust

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

        predictions = []
        trusters_cached = defaultdict(list)
        feedback_cached = defaultdict(list)
        isFeedbackADict = isinstance(self.feedback, dict)

        for i in range(0, len(users)):
            u = users[i]

            trustees_u = save_in_cache(self.trust, u, trusters_cached)

            if isFeedbackADict:
                feedback_u = self.feedback[u]
            else:
                feedback_u = save_in_cache(self.feedback, u, feedback_cached)

            pred = _predict(u, items[i], self.bias['globalAvg'],
                            self.bias['dUsers'], self.bias['dItems'], self.P,
                            self.Q, self.Y, self.W, feedback_u, trustees_u)
            predictions.append(pred[0])

        return super().predict_on_range(np.asarray(predictions))

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
        trusters_cached = defaultdict(list)
        feedback_cached = defaultdict(list)
        isFeedbackADict = isinstance(self.feedback, dict)

        for i in range(0, len(users)):
            u = users[i]

            trustees_u = save_in_cache(self.trust, u, trusters_cached)

            if isFeedbackADict:
                feedback_u = self.feedback[u]
            else:
                feedback_u = save_in_cache(self.feedback, u, feedback_cached)

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

    # def compute_objective(self, data, beta, beta_trust):
    #     data.X = data.X.astype(int)  # Convert indices to integer
    #
    #     users = data.X[:, self.order[0]]
    #     items = data.X[:, self.order[1]]
    #
    #     objective = _compute_objective(users, items,  self.bias['globalAvg'],
    #                                   self.bias['dUsers'], self.bias['dItems'],
    #                                   self.P, self.Q, self.Y, self.W,
    #                                   self.feedback, self.trust, beta,
    #                                    beta_trust)
    #     return objective

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

    def getWTable(self):
        domain_name = 'Trust-feature'
        variable = self.original_domain.variables[self.order[1]]
        return format_data.latent_factors_table(variable, self.W, domain_name)



if __name__ == "__main__":
    import Orange
    from sklearn.metrics import mean_squared_error

    print('Loading data...')
    ratings = Orange.data.Table('filmtrust/ratings.tab')
    trust = Orange.data.Table('filmtrust/trust.tab')

    start = time.time()
    learner = TrustSVDLearner(K=15, steps=1, alpha=0.07, beta=0.1,
                              beta_trust=0.05, trust=trust, verbose=True)
    recommender = learner(ratings)
    print('- Time (SVDPlusPlusLearner): %.3fs' % (time.time() - start))

    # prediction = recommender.predict(ratings, trust)
    # rmse = math.sqrt(mean_squared_error(ratings.Y, ))
    # print('- RMSE (SVDPlusPlusLearner): %.3f' % rmse)
