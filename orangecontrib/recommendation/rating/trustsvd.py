from orangecontrib.recommendation.rating import Learner, Model
from orangecontrib.recommendation.utils import format_data

import numpy as np
from scipy import sparse
import math

import time
import warnings

__all__ = ['TrustSVDLearner']


def _predict(users, items, global_avg, dUsers, dItems, P, Q, Y, W, feedback,
             trustees, subscripts='i,i'):
    bias = global_avg + dUsers[users] + dItems[items]

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
    p_enhanced = P[users, :] + (y_term + w_term)
    base_pred = np.einsum(subscripts, p_enhanced, Q[items, :])

    return bias + base_pred, y_term, w_term, norm_feedback, norm_trust


def _predict_all_items(users, global_avg, dUsers, dItems, P, Q):
    bias = global_avg + dUsers[users]
    tempB = np.tile(np.array(dItems), (len(users), 1))
    bias = bias[:, np.newaxis] + tempB

    base_pred = np.dot(P[users], Q.T)
    return bias + base_pred


def _compute_objective(users, items, global_avg, dUsers, dItems, P, Q, target,
                       beta):
    objective = 0
    subscripts = 'i,i'

    if len(users) > 1:
        subscripts = 'ij,ij->i'
    predictions = _predict(users, items, global_avg, dUsers, dItems, P, Q,
                           subscripts)
    objective += (predictions - target) ** 2

    # Regularization
    objective += beta * (np.linalg.norm(P[users, :], axis=1) ** 2
                         + np.linalg.norm(Q[items, :], axis=1) ** 2
                         + dItems[items] ** 2
                         + dUsers[users] ** 2)
    return objective.sum()


def _matrix_factorization(ratings, feedback, trust, bias, shape, order, K,
                          steps, alpha, beta, beta_trust, verbose=False,
                          random_state=None):
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
    P = np.random.rand(num_users, K)  # User and features
    Q = np.random.rand(num_items, K)  # Item and features
    Y = np.random.randn(num_users, K)
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

        # Optimize rating prediction
        for u, j in zip(*ratings.nonzero()):
            if verbose:
                start2 = time.time()

            items_rated_by_u = ratings[u, :].nonzero()[1]
            trustees_u = trust[u, :].nonzero()[1]
            users_who_rated_j = ratings[:, j].nonzero()[0]

            # Get indices of the items rated by u
            # if there is no feedback, infer from the ratings
            if feedback is None:
                feedback_u = items_rated_by_u
            else:
                feedback_u = feedback[u, :].nonzero()[1]

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

            if verbose:
                start2 = time.time()
            # Gradient Y
            if norm_feedback > 0:
                tempY1 = euj * (1/norm_feedback) * Q[j, :]
                for i in items_rated_by_u:
                    users_who_rated_i = ratings[:, i].nonzero()[0]
                    norm_Ui = math.sqrt(len(users_who_rated_i))
                    tempY[i, :] = tempY1 + (beta/norm_Ui) * Y[i, :]
            if verbose:
                print('\tTime iter ratings1: %.3fs' % (time.time() - start2))

            if verbose:
                start2 = time.time()
            # Gradient W
            if norm_trust > 0:
                tempW1 = euj * (1/norm_trust) * Q[j, :]  # W: Part 1
                for v in trustees_u:
                    trusters_v = trust[:, v].nonzero()[0]
                    norm_Tv = math.sqrt(len(trusters_v))
                    tempW[v, :] = tempW1 + (beta/norm_Tv) * W[v, :]
            if verbose:
                print('\tTime iter ratings2: %.3fs' % (time.time() - start2))

        # Optimize trust prediction
        for u, v in zip(*trust.nonzero()):
            if verbose:
                start2 = time.time()

            tuv_pred = np.dot(W[v, :], P[u, :])
            euv = tuv_pred - trust[u, v]

            # Get indices of the users who are trusted by u
            trustees_u = trust[u, :].nonzero()[1]
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
            print('\tTime: %.3fs' % (time.time() - start))

    return P, Q, Y, W


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
        max_user = np.max(data.X[:, self.order[0]])

        # Generate implicit feedback using the explicit ratings of the data
        # if self.feedback is None:
        #     self.feedback = format_data.generate_implicit_feedback(data,
        #                                                            self.order)

        # Transform rating matrix to sparse
        data = format_data.build_sparse_matrix(data.X[:, self.order[0]],
                                               data.X[:, self.order[1]],
                                               data.Y,
                                               self.shape)

        # Transform trust matrix to sparse
        max_row = np.max(self.trust.X[:, 0])
        max_col = np.max(self.trust.X[:, 1])
        max_user_t = int(max(max_user, max(max_row, max_col))) + 1
        shape_trust = (max_user_t, max_user_t)
        self.trust = sparse.csr_matrix((self.trust.Y,
                                        (self.trust.X[:, 0],
                                         self.trust.X[:, 1])),
                                       shape=shape_trust)

        # Factorize matrix
        self.P, self.Q, self.Y, self.W = \
            _matrix_factorization(ratings=data, feedback=self.feedback,
                                  trust=self.trust, bias=self.bias,
                                  shape=self.shape, order=self.order, K=self.K,
                                  steps=self.steps, alpha=self.alpha,
                                  beta=self.beta, beta_trust=self.beta_trust,
                                  verbose=self.verbose,
                                  random_state=self.random_state)

        model = TrustSVDModel(P=self.P, Q=self.Q, Y=self.Y, W=self.W,
                              bias=self.bias)
        return super().prepare_model(model)


class TrustSVDModel(Model):
    def __init__(self, P, Q, Y, W, bias):
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

        predictions = _predict_all_items(users, self.bias['globalAvg'],
                                         self.bias['dUsers'],
                                         self.bias['dItems'], self.P, self.Q)

        # Return top-k recommendations
        if top is not None:
            predictions = predictions[:, :top]

        return super().predict_on_range(predictions)

    def compute_objective(self, data, beta):
        data.X = data.X.astype(int)  # Convert indices to integer

        users = data.X[:, self.order[0]]
        items = data.X[:, self.order[1]]

        objective = _compute_objective(users, items, self.bias['globalAvg'],
                                       self.bias['dUsers'],
                                       self.bias['dItems'], self.P, self.Q,
                                       data.Y, beta)
        return objective

    def getPTable(self):
        variable = self.original_domain.variables[self.order[0]]
        return format_data.latent_factors_table(variable, self.P)

    def getQTable(self):
        variable = self.original_domain.variables[self.order[1]]
        return format_data.latent_factors_table(variable, self.Q)


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
