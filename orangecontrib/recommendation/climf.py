from Orange.base import Model, Learner
from Orange.evaluation.scoring import RMSE

from orangecontrib.recommendation.utils import format_data
from orangecontrib.recommendation.evaluation import MeanReciprocalRank

import numpy as np
from numpy import linalg as LA
import numpy.ma as ma
from scipy.special import expit as sigmoid

import math

import time
import warnings
import random

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

    name = 'CLiMF'

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



    def g(self, x):
        """sigmoid function"""
        return sigmoid(x)

    def dg(seld, x):
        ex = np.exp(-x)
        y = ex / (1 + ex) ** 2
        return y

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
        U = 0.01*np.random.rand(num_users, K)  # User and features
        V = 0.01*np.random.rand(num_items, K)  # Item and features

        if verbose:
            num_train_sample_users = min(num_users, 1000)
            train_sample_users = random.sample(range(num_users),
                                               num_train_sample_users)

        # Factorize matrix using SGD
        for step in range(steps):
            if verbose:
                start = time.time()
                print('- Step: %d' % (step+1))

            for i in range(len(U)):
                dU = -beta * U[i]

                # Precompute f (f[j] = <U[i], V[j]>)
                items = data.X[data.X[:, self.order[0]] == i][:, self.order[1]]
                f = np.einsum('j,ij->i', U[i], V[items])

                for j in range(len(items)):  #j=items
                    w = items[j]

                    dV = self.g(-f[j]) - beta * V[w]

                    # For I
                    vec1 = self.dg(f[j] - f) * \
                        (1/(1 - self.g(f - f[j])) - 1/(1 - self.g(f[j] - f)))
                    dV += np.einsum('i,j->ij', vec1, U[i]).sum(axis=0)

                    V[w] += alpha * dV
                    dU += self.g(-f[j]) * V[w]

                    # For II
                    vec2 = (V[items[j]] - V[items])
                    vec3 = self.dg(f - f[j])/(1 - self.g(f - f[j]))
                    dU += np.einsum('ij,i->ij', vec2, vec3).sum(axis=0)

                U[i] += alpha * dU

            if verbose:
                print('\tTime: %.3f' % (time.time() - start))
                print('\tMRR = {0:.4f}\n'.format(
                    self.compute_mrr(data.X, U,V, train_sample_users)))

        return U, V


    def compute_mrr(self, X, U, V, test_users=None):

        start = time.time()
        if test_users is None:
            test_users = range(len(U))

        # Get scores for all the items for a user[i]
        predictions = np.dot(U[test_users], V.T)
        predictions_sorted = np.argsort(predictions)
        predictions_sorted = np.fliplr(predictions_sorted)

        # Get items that are relevant for the user[i]
        all_items_u = []
        for i in test_users:
            items_u = X[X[:, self.order[0]] == i][:, self.order[1]]
            all_items_u.append(items_u)


        MRR = MeanReciprocalRank(predictions_sorted, all_items_u)

        print('\tTime MRR: %.3f' % (time.time() - start))
        return MRR


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

        # Check if all indices exist. If not, return random index.
        # On average, random indices is equivalent to return a global_average
        X = X[:, self.order[0]]
        X[X >= self.shape[0]] = np.random.randint(low=0, high=self.shape[0])

        # Compute scores
        predictions = np.dot(self.U[X], self.V.T)

        # Return indices of the sorted predictions
        predictions = np.argsort(predictions)
        predictions = np.fliplr(predictions)

        # Return top-k recommendations
        if top_k is not None:
            return predictions[:, :top_k]

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


    def __str__(self):
        return self.name