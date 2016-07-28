from orangecontrib.recommendation.ranking import Learner, Model
from orangecontrib.recommendation.utils import format_data

import numpy as np
from scipy.special import expit as sigmoid
from scipy.sparse import dok_matrix

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


def _matrix_factorization(data, shape, order, K, steps, alpha, beta,
                          verbose=False):
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

        verbose: boolean, optional
            If true, it outputs information about the process.

    Returns:
        U (matrix, UxK), V (matrix, KxI)

    """

    # Initialize factorized matrices randomly
    num_users, num_items = shape
    U = 0.01 * np.random.rand(num_users, K)  # User and features
    V = 0.01 * np.random.rand(num_items, K)  # Item and features

    user_col = order[0]
    item_col = order[1]
    # Factorize matrix using SGD
    for step in range(steps):
        if verbose:
            start = time.time()
            print('- Step: %d' % (step + 1))

        for i in range(len(U)):
            dU = -beta * U[i]

            # Precompute f (f[j] = <U[i], V[j]>)
            items = data.X[data.X[:, user_col] == i][:, item_col]
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

        if verbose:
            print('\tTime: %.3fs' % (time.time() - start))

    return U, V


class CLiMFLearner(Learner):
    """ Collaborative Less-is-More Filtering Matrix Factorization

    Matrix factorization for scenarios with binary relevance data when only a
    few (k) items are recommended to individual users. It improves top-k
    recommendations through ranking by directly maximizing the Mean Reciprocal
    Rank (MRR).


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

        # Factorize matrix
        self.U, self.V = _matrix_factorization(data=data, shape=self.shape,
                                               order=self.order, K=self.K,
                                               steps=self.steps,
                                               alpha=self.alpha,
                                               beta=self.beta, verbose=False)

        model = CLiMFModel(U=self.U, V=self.V)
        return super().prepare_model(model)


class CLiMFModel(Model):

    def __init__(self, U, V):
        """This model receives a learner and provides and interface to make the
        predictions for a given user.

        Args:
            U: Matrix (users x Latent_factors)

            V: Matrix (items x Latent_factors)

            order: (int, int)
                Tuple with the index of the columns users and items in X. (idx_user, idx_item)

       """
        self.U = U
        self.V = V

    def predict(self, X, top_k=None):
        """This function returns all the predictions for a set of items.
        If users is set to 'None', it will return all the predictions for all
        the users (matrix of size [num_users x num_items]).

        Args:
            users: array, optional
                Array with the indices of the users to which make the
                predictions.

            top_k: int, optional
                Return just the top k recommendations.

        Returns:
            Array with the indices of the items recommended, sorted by ascending
            ranking. (1st better than 2nd, than 3rd,...)

        """

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
        return format_data.latent_factors_table(variable, self.U)

    def getVTable(self):
        variable = self.original_domain.variables[self.order[1]]
        return format_data.latent_factors_table(variable, self.V)
