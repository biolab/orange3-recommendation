from orangecontrib.recommendation.ranking import Learner, Model
from orangecontrib.recommendation.utils.format_data import *
from orangecontrib.recommendation.utils.datacaching import cache_rows

from collections import defaultdict

from scipy.special import expit as sigmoid
from scipy.sparse import dok_matrix

import numpy as np
import time
import warnings

__all__ = ['CLiMFLearner']
__sparse_format__ = lil_matrix


def _g(x):
    """sigmoid function"""
    return sigmoid(x)


def _dg(x):
    ex = np.exp(-x)
    y = ex / (1 + ex) ** 2
    return y


def _matrix_factorization(ratings, shape, num_factors, num_iter, learning_rate,
                          lmbda, verbose=False, random_state=None):
    # Seed the generator
    if random_state is not None:
        np.random.seed(random_state)

    # Get featured matrices dimensions
    num_users, num_items = shape

    # Initialize low-rank matrices
    U = 0.01 * np.random.rand(num_users, num_factors)  # User-feature matrix
    V = 0.01 * np.random.rand(num_items, num_factors)  # Item-feature matrix

    # Cache rows
    users_cached = defaultdict(list)

    # Factorize matrix using SGD
    for step in range(num_iter):
        if verbose:
            start = time.time()
            print('- Step: %d' % (step + 1))

        # Optimize rating prediction
        objective = 0
        for i in range(len(U)):
            dU = -lmbda * U[i]

            # Precompute f (f[j] = <U[i], V[j]>)
            items = cache_rows(ratings, i, users_cached)
            f = np.einsum('j,ij->i', U[i], V[items])

            for j in range(len(items)):  # j=items
                w = items[j]

                dV = _g(-f[j]) - lmbda * V[w]

                # For I
                vec1 = _dg(f[j] - f) * \
                       (1 / (1 - _g(f - f[j])) - 1 / (1 - _g(f[j] - f)))
                dV += np.einsum('i,j->ij', vec1, U[i]).sum(axis=0)

                V[w] += learning_rate * dV
                dU += _g(-f[j]) * V[w]

                # For II
                vec2 = (V[items[j]] - V[items])
                vec3 = _dg(f - f[j]) / (1 - _g(f - f[j]))
                dU += np.einsum('ij,i->ij', vec2, vec3).sum(axis=0)

            U[i] += learning_rate * dU

            # TODO: Loss function

        # Print process
        if verbose:
            if verbose > 1:
                print('\t- Loss: %.3f' % objective)
            print('\t- Time: %.3fs' % (time.time() - start))
            print('')

    return U, V


class CLiMFLearner(Learner):
    """CLiMF: Collaborative Less-is-More Filtering Matrix Factorization

    This model uses stochastic gradient descent to find two low-rank
    matrices: user-feature matrix and item-feature matrix.

    CLiMF is a matrix factorization for scenarios with binary relevance data
    when only a few (k) items are recommended to individual users. It improves top-k
    recommendations through ranking by directly maximizing the Mean Reciprocal
    Rank (MRR).


    Attributes:
        num_factors: int, optional
            The number of latent factors.

        num_iter: int, optional
            The number of passes over the training data (aka epochs).

        learning_rate: float, optional
            The learning rate controlling the size of update steps (general).

        lmbda: float, optional
            Controls the importance of the regularization term (general).
            Avoids overfitting by penalizing the magnitudes of the parameters.

        verbose: boolean or int, optional
            Prints information about the process according to the verbosity
            level. Values: False (verbose=0), True (verbose=1) and INTEGER

        random_state: int, optional
            Set the seed for the numpy random generator, so it makes the random
            numbers predictable. This a debbuging feature.
    """

    name = 'CLiMF'

    def __init__(self, num_factors=5, num_iter=25, learning_rate=0.07,
                 lmbda=0.1, preprocessors=None, verbose=False,
                 random_state=None):
        self.num_factors = num_factors
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.lmbda = lmbda
        self.random_state = random_state

        super().__init__(preprocessors=preprocessors, verbose=verbose)

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

        # Transform ratings matrix into a sparse matrix
        data = table2sparse(data, self.shape, self.order,
                            m_type=__sparse_format__)

        # Factorize matrix
        U, V = _matrix_factorization(ratings=data, shape=self.shape,
                                     num_factors=self.num_factors,
                                     num_iter=self.num_iter,
                                     learning_rate=self.learning_rate,
                                     lmbda=self.lmbda, verbose=self.verbose,
                                     random_state=self.random_state)

        # Construct model
        model = CLiMFModel(U=U, V=V)
        return super().prepare_model(model)


class CLiMFModel(Model):

    def __init__(self, U, V):
        self.U = U
        self.V = V
        super().__init__()

    def predict(self, X, top_k=None):
        """Perform predictions on samples in X for all items.

        Args:
            X: array, optional
                Array with the indices of the users to which make the
                predictions. If None (default), predicts for all users.

            top_k: int, optional
                Returns the k-first predictions. (Do not confuse with
                'top-best').

        Returns:
            C: ndarray, shape = (n_samples, n_items)
                Returns predicted values. A matrix (U, I) with the indices of
                the items recommended, sorted by ascending ranking. (1st better
                than 2nd, than 3rd,...)

        """

        # Compute scores
        predictions = np.dot(self.U[X], self.V.T)

        # Return indices of the sorted predictions
        predictions = np.argsort(predictions)
        predictions = np.fliplr(predictions)

        # Return top-k recommendations
        if top_k is not None:
            predictions = predictions[:, :top_k]

        return predictions

    def compute_objective(self, X, Y, U, V, lmbda):
        # TODO: Cast rows, cols through preprocess
        # X and Y are original data
        # Construct explicit sparse matrix to evaluate the objective function
        M, N = U.shape[0], V.shape[0]
        Ys = dok_matrix((M, N))
        for (i, j), y in zip(X, Y):
            Ys[i, j] = y
        Ys = Ys.tocsr()

        W1 = np.log(sigmoid(U.dot(V.T)))
        W2 = np.zeros(Ys.shape)

        for i in range(M):
            for j in range(N):
                W2[i, j] = sum((np.log(
                    1 - Ys[i, k] * sigmoid(U[i, :].dot(V[k, :] - V[j, :])))
                                for k in range(N)))

        objective = Ys.multiply(W1 - W2).sum()
        objective -= lmbda / 2.0 * (np.linalg.norm(U, ord="fro") ** 2
                                        + np.linalg.norm(V, ord="fro") ** 2)

        return objective

    def getUTable(self):
        variable = self.original_domain.variables[self.order[0]]
        return feature_matrix(variable, self.U)

    def getVTable(self):
        variable = self.original_domain.variables[self.order[1]]
        return feature_matrix(variable, self.V)
