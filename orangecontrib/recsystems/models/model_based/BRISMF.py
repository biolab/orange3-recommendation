from Orange.base import Model, Learner

import numpy as np
from numpy import linalg as LA

from scipy import sparse

import math

import warnings
import time

__all__ = ['BRISMFLearner']

class BRISMFLearner(Learner):
    """ Biased Regularized Incremental Simultaneous Matrix Factorization

    This model uses stochastic gradient descent to find the ratings of two
    low-rank matrices which represents user and item factors. This object can
    factorize either dense or sparse matrices.

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
        self.P = None
        self.Q = None
        self.bias = None
        self.verbose = verbose

        super().__init__(preprocessors=preprocessors)
        self.params = vars()


    def fit(self, X, Y=None, W=None):
        """This function calls the factorization method.

        Args:
            X: Matrix
                Data to fit.

        Returns:
            Model object (BRISMFModel).

        """
        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well. You are advised to use the LinearRegression "
                          "estimator", stacklevel=2)

        # Optional, can be manage through preprocessors.
        X = self.prepare_data(X)

        # Factorize matrix
        self.P, self.Q, self.bias = self.matrix_factorization(
                                                                X,
                                                                self.K,
                                                                self.steps,
                                                                self.alpha,
                                                                self.beta,
                                                                self.verbose)
        return BRISMFModel(P=self.P, Q=self.Q, bias=self.bias)

    def prepare_data(self, X):
        """Function to remove NaNs from the data (preprocessor)

        Args:
            X: Matrix (data to fit).

        Returns:
            X (matrix)

        """

        # Convert NaNs to zero
        where_are_NaNs = np.isnan(X)
        X[where_are_NaNs] = 0

        # Transform dense matrix into sparse matrix
        #X = sparse.csr_matrix(X)

        return X

    def matrix_factorization(self, R, K, steps, alpha, beta, verbose=False):
        """ Factorize either a dense matrix or a sparse matrix into two low-rank
         matrices which represents user and item factors.

        Args:
            R: Matrix
                Matrix to factorize. (Zeros are equivalent to unknown data)

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
            , 'delta users', 'global mean items' and 'global mean users')

        """

        # Initialize factorized matrices randomly
        num_users, num_items = R.shape
        P = np.random.rand(num_users, K)  # User and features
        Q = np.random.rand(num_items, K)  # Item and features


        # Check if R is a sparse matrix
        if isinstance(R, sparse.csr_matrix) or \
                isinstance(R, sparse.csc_matrix):
            start2 = time.time()
            # Local means (array)
            mean_user_rating = np.ravel(R.mean(axis=1))  # Rows
            mean_item_rating = np.ravel(R.mean(axis=0))  # Columns

            # Global mean
            global_mean_users = mean_user_rating.mean()
            global_mean_items = mean_item_rating.mean()

            if verbose:
                print('- Time mean (sparse): %.3fs\n' % (time.time() - start2))

        else:  # Dense matrix
            start2 = time.time()
            # Local means (array)
            mean_user_rating = np.mean(R, axis=1)  # Rows
            mean_item_rating = np.mean(R, axis=0)  # Columns

            # Global mean
            global_mean_users = np.mean(mean_user_rating)
            global_mean_items = np.mean(mean_item_rating)

            if verbose:
                print('- Time mean (dense): %.3fs\n' % (time.time() - start2))


        # Compute bias and deltas (Common - Dense/Sparse matrices)
        deltaUser = mean_user_rating - global_mean_users
        deltaItem = mean_item_rating - global_mean_items
        bias = {'dItems': deltaItem,
                'dUsers': deltaUser,
                'gMeanItems': global_mean_items,
                'gMeanUsers': global_mean_users}

        # Get non-zero elements
        #indices = R.nonzero()
        indices = np.array(np.nonzero(R > 0)).T

        # Factorize matrix using SGD
        for step in range(steps):

            # Compute predictions
            for i, j in indices:

                    # Try to remove this (try indexing)
                    if R[i, j] > 0:  # This makes sparse matrices really slow
                        #masked

                        rij_pred = global_mean_items + \
                                   deltaItem[j] + \
                                   deltaUser[i] + \
                                   np.dot(P[i, :], Q[j, :])

                        eij = rij_pred - R[i, j]

                        tempP = alpha * 2 * (eij * Q[j] + beta * LA.norm(P[i]))
                        tempQ = alpha * 2 * (eij * P[i] + beta * LA.norm(Q[j]))
                        P[i] -= tempP
                        Q[j] -= tempQ

        # Compute error (this section can be commented)
        if verbose:
            counter = 0
            error = 0
            for i in range(0, num_users):
                for j in range(0, num_items):
                    if R[i, j] > 0:
                        counter +=1
                        rij_pred = global_mean_items + \
                                   deltaItem[j] + \
                                   deltaUser[i] + \
                                   np.dot(P[i, :], Q[j, :])
                        error += (rij_pred - R[i, j])**2

            error = math.sqrt(error/counter)
            print('- RMSE: %.3f' % error)

        return P, Q, bias


class BRISMFModel(Model):

    def __init__(self, P, Q, bias):
        """This model receives a learner and provides and interface to make the
        predictions for a given user.

        Args:
            P: Matrix
            Q: Matrix
            bias: dictionary
                'delta items', 'delta users', 'global mean items' and
                'global mean users'

       """
        self.P = P
        self.Q = Q
        self.bias = bias


    # Predict top-best items for a user
    def predict(self, indices):
        """This function receives the index of a user and returns its
        recomendations.

        Args:
            user: int
                Index of the user to which make the predictions.

            sort: boolean, optional
                If True, the returned array with the ratings will be sorted in
                descending order.

            top: int, optional
                Return just the first k recommendations.

        Returns:
            Array with the recommendations for a given user.

        """

        #indices = np.array(np.nonzero(tuples > 0)).T
        asd = 2
        bias = self.bias['gMeanItems'] + \
                    self.bias['dUsers'][indices[:, 0]] + \
                    self.bias['dItems'][indices[:, 1]]

        tempP = self.P[indices[:, 0]]
        tempQ = self.Q[indices[:, 1]]

        #base_pred = np.multiply(tempP, tempQ)
        base_pred = np.einsum('ij,ij->i', tempP, tempQ)
        predictions = bias + base_pred

        return predictions

    # Predict top-best items for a user
    def predict_items(self, user, sort=True, top=None):
        """This function receives the index of a user and returns its
        recomendations.

        Args:
            user: int
                Index of the user to which make the predictions.

            sort: boolean, optional
                If True, the returned array with the ratings will be sorted in
                descending order.

            top: int, optional
                Return just the first k recommendations.

        Returns:
            Array with the recommendations for a given user.

        """
        bias = self.bias['gMeanItems'] + \
               self.bias['dUsers'][user] + \
               self.bias['dItems']
        base_pred = np.dot(self.P[user], self.Q.T)
        predictions = bias + base_pred

        # Sort predictions
        if sort:
            indices = np.argsort(predictions)[::-1]  # Descending order
        else:
            indices = np.arange(0, len(predictions))

        # Join predictions and indices
        predictions = np.array((indices, predictions[indices])).T

        # Return top-k recommendations
        if top != None:
            return predictions[:top]

        return predictions


    def __str__(self):
        return 'BRISMF {}'.format('--> return model')