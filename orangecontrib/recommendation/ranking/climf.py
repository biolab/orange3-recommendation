from orangecontrib.recommendation.rating import Learner, Model
from orangecontrib.recommendation.utils.format_data import feature_matrix

from scipy.special import expit as sigmoid
from scipy.sparse import dok_matrix

import numpy as np
import time
import warnings

__all__ = ['CLiMFLearner']


def _g(x):
    """sigmoid function"""
    return sigmoid(x)


def _dg(x):
    ex = np.exp(-x)
    y = ex / (1 + ex) ** 2
    return y


def _matrix_factorization(ratings, shape, order, K, steps, alpha, beta,
                          verbose=False, random_state=None):
    # Seed the generator
    if random_state is not None:
        np.random.seed(random_state)

    # Get featured matrices dimensions
    num_users, num_items = shape

    # Initialize low-rank matrices
    U = 0.01 * np.random.rand(num_users, K)  # User-feature matrix
    V = 0.01 * np.random.rand(num_items, K)  # Item-feature matrix

    # Get positional index of base columns
    user_col, item_col = order

    # Factorize matrix using SGD
    for step in range(steps):
        if verbose:
            start = time.time()
            print('- Step: %d' % (step + 1))

        # Optimize rating prediction
        objective = 0
        for i in range(len(U)):
            dU = -beta * U[i]

            # Precompute f (f[j] = <U[i], V[j]>)
            items = ratings.X[ratings.X[:, user_col] == i][:, item_col]
            f = np.einsum('j,ij->i', U[i], V[items])

            for j in range(len(items)):  # j=items
                w = items[j]

                dV = _g(-f[j]) - beta * V[w]

                # For I
                vec1 = _dg(f[j] - f) * \
                       (1 / (1 - _g(f - f[j])) - 1 / (1 - _g(f[j] - f)))
                dV += np.einsum('i,j->ij', vec1, U[i]).sum(axis=0)

                V[w] += alpha * dV
                dU += _g(-f[j]) * V[w]

                # For II
                vec2 = (V[items[j]] - V[items])
                vec3 = _dg(f - f[j]) / (1 - _g(f - f[j]))
                dU += np.einsum('ij,i->ij', vec2, vec3).sum(axis=0)

            U[i] += alpha * dU

            # TODO: Loss function

        if verbose:
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
        K: int, optional
            The number of latent factors.

        steps: int, optional
            The number of passes over the training data (aka epochs).

        alpha: float, optional
            The learning rate.

        beta: float, optional
            The regularization for the ratings.

        verbose: boolean, optional
            Prints information about the process.

        random_state: int, optional
            Set the seed for the numpy random generator, so it makes the random
            numbers predictable. This a debbuging feature.
    """

    name = 'CLiMF'

    def __init__(self, K=5, steps=25, alpha=0.07, beta=0.1, preprocessors=None,
                 verbose=False):
        self.K = K
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.U = None
        self.V = None
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
        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well.", stacklevel=2)

        # Factorize matrix
        self.U, self.V = _matrix_factorization(ratings=data, shape=self.shape,
                                               order=self.order, K=self.K,
                                               steps=self.steps,
                                               alpha=self.alpha,
                                               beta=self.beta, verbose=False)

        # Construct model
        model = CLiMFModel(U=self.U, V=self.V)
        return super().prepare_model(model)


class CLiMFModel(Model):

    def __init__(self, U, V):
        self.U = U
        self.V = V

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

        # Prepare data
        super().prepare_predict(X)

        # Compute scores
        predictions = np.dot(self.U[X], self.V.T)

        # Return indices of the sorted predictions
        predictions = np.argsort(predictions)
        predictions = np.fliplr(predictions)

        # Return top-k recommendations
        if top_k is not None:
            predictions = predictions[:, :top_k]

        return predictions

    def compute_objective(self, X, Y, U, V, beta):
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
        objective -= beta / 2.0 * (np.linalg.norm(U, ord="fro") ** 2
                                        + np.linalg.norm(V, ord="fro") ** 2)

        return objective

    def getUTable(self):
        variable = self.original_domain.variables[self.order[0]]
        return feature_matrix(variable, self.U)

    def getVTable(self):
        variable = self.original_domain.variables[self.order[1]]
        return feature_matrix(variable, self.V)
