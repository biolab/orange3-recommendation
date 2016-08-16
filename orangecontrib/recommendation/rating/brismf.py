from orangecontrib.recommendation.rating import Learner, Model
from orangecontrib.recommendation.utils.format_data import *
from orangecontrib.recommendation.utils.sgd_optimizer import *

import numpy as np
import time
import warnings
import copy

__all__ = ['BRISMFLearner']
__sparse_format__ = lil_matrix


def _predict(users, items, global_avg, bu, bi, P, Q, subscripts='i,i'):
    bias = global_avg + bu[users] + bi[items]
    base_pred = np.einsum(subscripts, P[users, :], Q[items, :])
    return bias + base_pred


def _predict_all_items(users, global_avg, bu, bi, P, Q):
    bias = global_avg + bu[users]
    tempB = np.tile(np.array(bi), (len(users), 1))
    bias = bias[:, np.newaxis] + tempB

    base_pred = np.dot(P[users], Q.T)
    return bias + base_pred


def _matrix_factorization(ratings, bias, shape, num_factors, num_iter,
                          learning_rate, bias_learning_rate, lmbda, bias_lmbda,
                          optimizer, verbose=False, random_state=None):
    # Seed the generator
    if random_state is not None:
        np.random.seed(random_state)

    # Get featured matrices dimensions
    num_users, num_items = shape

    # Initialize low-rank matrices
    P = np.random.rand(num_users, num_factors)  # User-feature matrix
    Q = np.random.rand(num_items, num_factors)  # Item-feature matrix

    # Compute bias (not need it if learnt)
    global_avg = bias['globalAvg']
    bu = bias['dUsers']
    bi = bias['dItems']

    # Configure optimizer
    update_bu = create_opt(optimizer, bias_learning_rate).update
    update_bj = create_opt(optimizer, bias_learning_rate).update
    update_pu = create_opt(optimizer, learning_rate).update
    update_qj = create_opt(optimizer, learning_rate).update

    # Print information about the verbosity level
    if verbose:
        print('BRISMF factorization started.')
        print('\tLevel of verbosity: ' + str(int(verbose)))
        print('\t\t- Verbosity = 1\t->\t[time/iter]')
        print('\t\t- Verbosity = 2\t->\t[time/iter, loss]')
        print('')

    # Factorize matrix using SGD
    for step in range(num_iter):
        if verbose:
            start = time.time()
            print('- Step: %d' % (step + 1))


        # Optimize rating prediction
        for u, j in zip(*ratings.nonzero()):

            # Prediction and error
            rij_pred = _predict(u, j, global_avg, bu, bi, P, Q)
            eij = ratings[u, j] - rij_pred

            # Compute gradients
            dx_bu = -eij + bias_lmbda * bu[u]
            dx_bi = -eij + bias_lmbda * bi[j]
            dx_pu = -eij * Q[j, :] + lmbda * P[u, :]
            dx_qi = -eij * P[u, :] + lmbda * Q[j, :]

            # Update the gradients at the same time
            update_bu(dx_bu, bu, u)
            update_bj(dx_bi, bi, j)
            update_pu(dx_pu, P, u)
            update_qj(dx_qi, Q, j)

        # Print process
        if verbose:
            print('\t- Time: %.3fs' % (time.time() - start))

            if verbose > 1:
                # Set parameters and compute loss
                bias = (global_avg, bu, bi)
                low_rank_matrices = (P, Q)
                params = (lmbda, bias_lmbda)
                objective = compute_loss(ratings, bias, low_rank_matrices, params)
                print('\t- Training loss: %.3f' % objective)
            print('')

    return P, Q, bu, bi


def compute_loss(data, bias, low_rank_matrices, params):

    # Set parameters
    ratings = data
    global_avg, bu, bi = bias
    P, Q = low_rank_matrices
    lmbda, bias_lmbda = params

    # Check data type
    if isinstance(ratings, __sparse_format__):
        pass
    elif isinstance(ratings, Table):
        # Preprocess Orange.data.Table and transform it to sparse
        ratings, order, shape = preprocess(ratings)
        ratings = table2sparse(ratings, shape, order, m_type=__sparse_format__)
    else:
        raise TypeError('Invalid data type')

    # Compute loss
    objective = 0
    for u, j in zip(*ratings.nonzero()):
        ruj_pred = _predict(u, j, global_avg, bu, bi, P, Q)
        objective += (ratings[u, j] - ruj_pred) ** 2  # error^2

        # Regularization
        objective += lmbda * (np.linalg.norm(P[u, :]) ** 2 +
                              np.linalg.norm(Q[j, :]) ** 2) + \
                     bias_lmbda * (bu[u] ** 2 + bi[j] ** 2)
    return objective


class BRISMFLearner(Learner):
    """BRISMF: Biased Regularized Incremental Simultaneous Matrix Factorization

    This model uses stochastic gradient descent to find two low-rank
    matrices: user-feature matrix and item-feature matrix.

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

        optimizer: Optimizer, optional
            Set the optimizer for SGD. If None (default), classical SGD will be
            applied.

        verbose: boolean or int, optional
            Prints information about the process according to the verbosity
            level. Values: False (verbose=0), True (verbose=1) and INTEGER

        random_state: int, optional
            Set the seed for the numpy random generator, so it makes the random
            numbers predictable. This a debbuging feature.

    """

    name = 'BRISMF'

    def __init__(self, num_factors=5, num_iter=25, learning_rate=0.07,
                 bias_learning_rate=None, lmbda=0.1, bias_lmbda=None,
                 min_rating=None, max_rating=None, optimizer=None,
                 preprocessors=None, verbose=False, random_state=None):
        self.num_factors = num_factors
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.bias_learning_rate = bias_learning_rate
        self.lmbda = lmbda
        self.bias_lmbda = bias_lmbda
        self.optimizer = SGD() if optimizer is None else optimizer
        self.random_state = random_state

        # Correct assignments
        if self.bias_learning_rate is None:
            self.bias_learning_rate = self.learning_rate
        if self.bias_lmbda is None:
            self.bias_lmbda = self.lmbda

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
        P, Q, bu, bi = _matrix_factorization(ratings=data, bias=bias,
                                     shape=self.shape,
                                     num_factors=self.num_factors,
                                     num_iter=self.num_iter,
                                     learning_rate=self.learning_rate,
                                     bias_learning_rate=self.bias_learning_rate,
                                     lmbda=self.lmbda,
                                     bias_lmbda=self.bias_lmbda,
                                     optimizer=self.optimizer,
                                     verbose=self.verbose,
                                     random_state=self.random_state)

        # Update biases
        bias['dUsers'] = bu
        bias['dItems'] = bi

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

    def getPTable(self):
        variable = self.original_domain.variables[self.order[0]]
        return feature_matrix(variable, self.P)

    def getQTable(self):
        variable = self.original_domain.variables[self.order[1]]
        return feature_matrix(variable, self.Q)
