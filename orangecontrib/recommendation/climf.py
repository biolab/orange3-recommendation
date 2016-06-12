from Orange.base import Model, Learner
from orangecontrib.recommendation.utils import format_data

import numpy as np
from numpy import linalg as LA

import math

import warnings

__all__ = ['CLiMFLearner']

class CLiMFLearner(Learner):
    """ Collaborative Less-is-More Filtering Matrix Factorization

    This model is focused on improving top-k recommendations through
    ranking by directly maximizing the Mean Reciprocal Rank (MRR)

    Attributes:
        K: int, optional
            The number of latent factors.

        steps: int, optional (default = 100)
            The number of epochs of stochastic gradient descent.

        alpha: float, optional (default = 0.005)
            The learning rate of stochastic gradient descent.

        beta: float, optional (default = 0.02)
            The regularization parameter.

        verbose: boolean, optional (default = False)
            Prints information about the process.
    """

    name = 'BRISMF'

    def __init__(self,
                 K=2,
                 steps=100,
                 alpha=0.005,
                 beta=0.02,
                 preprocessors=None,
                 verbose=False):
        self.K = K
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.U = None
        self.V = None
        self.verbose = verbose
        self.shape = None
        self.order = None

        super().__init__(preprocessors=preprocessors)
        self.params = vars()


    def fit_storage(self, data):
        """This function calls the factorization method.

        Args:
            data: Orange.data.Table

        Returns:
            Model object (BRISMFModel).

        """


        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well. You are advised to use the LinearRegression "
                          "estimator", stacklevel=2)

        # Optional, can be manage through preprocessors.
        data, self.order, self.shape = format_data.preprocess(data)

        # Factorize matrix
        self.U, self.V = self.matrix_factorization(data,
                                                    self.K,
                                                    self.steps,
                                                    self.alpha,
                                                    self.beta,
                                                    self.verbose)


        return CLiMFModel(U=self.U,
                          V=self.V,
                           order=self.order)

    def tensor(self, x, beta):
        return pow(math.e, -beta * x)

    def g(self, x):
        """sigmoid function"""
        return 1 / (1 + math.exp(-x))

    def dg(self, x):
        """derivative of sigmoid function"""
        return math.exp(x) / (1 + math.exp(x)) ** 2

    def precompute_f(self, X, U, V, i):
        """precompute f[j] = <U[i],V[j]>
        params:
          data: scipy csr sparse matrix containing user->(item,count)
          U   : user factors
          V   : item factors
          i   : item of interest
        returns:
          dot products <U[i],V[j]> for all j in data[i]
        """
        items = X[i]
        f = dict((j, np.dot(U[i], V[j])) for j in items)
        return f


    def matrix_factorization(self, data, K, steps, alpha, beta, verbose=False):
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
            P (matrix, UxK), Q (matrix, KxI) and bias (dictionary, 'delta items'
            , 'delta users')

        """

        # Initialize factorized matrices randomly
        num_users, num_items = self.shape
        U = np.random.rand(num_users, K)  # User and features
        V = np.random.rand(num_items, K)  # Item and features

        # Factorize matrix using SGD
        for step in range(steps):
            if verbose:
                print('- Step: %d' % (step+1))

            for i in range(len(U)):
                dU = -beta * U[i]
                f = self.precompute_f(data.X, U, V, i)
                for j in f:
                    dV = self.g(-f[j]) - beta * V[j]
                    for k in f:
                        inv_g1 = 1 - self.g(f[k] - f[j])
                        inv_g2 = 1 - self.g(f[j] - f[k])

                        dV += self.dg(f[j] - f[k]) * (
                        1 / (inv_g1) - 1 / (inv_g2)) * U[i]
                    V[j] += alpha * dV
                    dU += self.g(-f[j]) * V[j]
                    for k in f:
                        dU += (V[j] - V[k]) * self.dg(f[k] - f[j]) / (
                        1 - self.g(f[k] - f[j]))
                U[i] += alpha * dU

        return U, V





class CLiMFModel(Model):

    def __init__(self, U, V, order):
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
        self.shape = (len(self.U), len(self.V))
        self.order = order


    def predict(self, X):
        """This function receives an array of indexes [[idx_user, idx_item]] and
        returns the prediction for these pairs.

            Args:
                X: Matrix (2xN)
                    Matrix that contains pairs of the type user-item

            Returns:
                Array with the recommendations for a given user.

            """

        # Check if all indices exist. If not, return random index.
        # On average, random indices is equivalent to return a global_average
        X[X[:, self.order[0]] >= self.shape[0], self.order[0]] = \
            np.random.randint(low=0, high=self.shape[0])
        X[X[:, self.order[1]] >= self.shape[1], self.order[1]] = \
            np.random.randint(low=0, high=self.shape[1])

        tempU = self.U[X[:, self.order[0]]]
        tempV = self.V[X[:, self.order[1]]]

        predictions = np.einsum('ij,ij->i', tempU, tempV)

        return predictions


    def predict_storage(self, data):
        """ Convert data.X variables to integer and calls predict(data.X)

        Args:
            data: Orange.data.Table

        Returns:
            Array with the recommendations for a given user.

        """

        # Convert indices to integer and call predict()
        return self.predict(data.X.astype(int))


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
            users = np.asarray(range(0, self.shape[0]))

        predictions = np.einsum('ij,ij->i', self.U[users], self.V)

        # Return top-k recommendations
        if top is not None:
            return predictions[:, :top]

        return predictions


    def __str__(self):
        return self.name