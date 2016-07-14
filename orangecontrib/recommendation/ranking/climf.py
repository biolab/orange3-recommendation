from Orange.data import Table, Domain, ContinuousVariable, StringVariable

from orangecontrib.recommendation import Learner, Model
from orangecontrib.recommendation.utils import format_data

import numpy as np
from scipy.special import expit as sigmoid

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


    def fit_model(self, data):
        """This function calls the factorization method.

        Args:
            data: Orange.data.Table

        Returns:
            Model object (BRISMFModel).

        """

        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well.", stacklevel=2)

        # Factorize matrix
        self.U, self.V = _matrix_factorization(data=data, shape=self.shape,
                                               order=self.order, K=self.K,
                                               steps=self.steps,
                                               alpha=self.alpha,
                                               beta=self.beta, verbose=False)

        return CLiMFModel(U=self.U, V=self.V)



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

    def __call__(self, *args, **kwargs):
        """
        We need to override the __call__ of the base.model because it transforms
        the output to 'argmax(probabilities=X)'
        """

        data = args[0]
        top_k = None
        if 'top_k' in kwargs:  # Check if this parameters exists
            top_k = kwargs['top_k']

        if isinstance(data, np.ndarray):
            prediction = self.predict(X=data, top_k=top_k)
        elif isinstance(data, Table):
            prediction = self.predict(X=data.X.astype(int), top_k=top_k)
        else:
            raise TypeError("Unrecognized argument (instance of '{}')"
                            .format(type(data).__name__))

        return prediction

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

        # Compute scores
        predictions = np.dot(self.U[X], self.V.T)

        # Return indices of the sorted predictions
        predictions = np.argsort(predictions)
        predictions = np.fliplr(predictions)

        # Return top-k recommendations
        if top_k is not None:
            predictions = predictions[:, :top_k]

        return predictions

    def getUTable(self):
        variable = self.original_domain.variables[self.order[0]]
        return format_data.latent_factors_table(variable, self.U)

    def getVTable(self):
        variable = self.original_domain.variables[self.order[1]]
        return format_data.latent_factors_table(variable, self.V)
