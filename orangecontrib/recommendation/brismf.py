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

    This model uses stochastic gradient descent to find the values of two
    low-rank matrices which represents the user and item factors. This object
    can factorize either dense or sparse matrices.

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
        self.P = None
        self.Q = None
        self.bias = None
        self.global_average = None
        self.verbose = verbose
        self.shape = None

        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def format_data(self, data):
        """Transforms the raw data read by Orange into something that this
        class can use

        Args:
            data: Orange.data.Table

        Returns:
            data

        """

        col_attributes = [a for a in data.domain.attributes + data.domain.metas
                          if a.attributes.get("col")]

        col_attribute = col_attributes[0] if len(
            col_attributes) == 1 else print("warning")

        row_attributes = [a for a in data.domain.attributes + data.domain.metas
                          if a.attributes.get("row")]

        row_attribute = row_attributes[0] if len(
            row_attributes) == 1 else print("warning")

        # Get indices of the columns
        idx_items = data.domain.variables.index(col_attribute)
        idx_users = data.domain.variables.index(row_attribute)

        users = len(data.domain.variables[idx_users].values)
        items = len(data.domain.variables[idx_items].values)
        self.shape = (users, items)


        # ***** In the case of using 'strings' instead of 'discrete'
        # Get values of the columns
        # users = data.metas[:, idx_users]
        # items = data.metas[:, idx_items]
        #
        # # Remove repeated elements
        # set_users = set(users)
        # set_items = set(items)
        # shape = (len(set_users), len(set_items))
        #
        # # Build dictionary to know the indices of the key
        # dict_users = dict(zip(set_users, range(0, shape[0])))
        # dict_items = dict(zip(set_items, range(0, shape[1])))
        #
        # col_indices_users = [dict_users[x] for x in users]
        # col_indices_items = [dict_items[x] for x in items]
        #
        # data.X = np.column_stack((col_indices_users, col_indices_items))

        # Convert to integer
        data.X = data.X.astype(int)

        return data


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
        data = self.format_data(data)

        # Compute global average
        self.global_average = np.mean(data.Y)

        # Factorize matrix
        self.P, self.Q, self.bias = self.matrix_factorization(data,
                                                                self.K,
                                                                self.steps,
                                                                self.alpha,
                                                                self.beta,
                                                                self.verbose)


        return BRISMFModel(P=self.P,
                           Q=self.Q,
                           bias=self.bias,
                           global_average=self.global_average)




    def matrix_factorization(self, data, K, steps, alpha, beta, verbose=False):
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
        num_users, num_items = self.shape
        P = np.random.rand(num_users, K)  # User and features
        Q = np.random.rand(num_items, K)  # Item and features

        # Compute biases
        bias = self.compute_bias(data)

        # Factorize matrix using SGD
        for step in range(steps):

            # Compute predictions
            for k in range(0, len(data.Y)):
                i, j = data.X[k]

                rij_pred = self.global_average + \
                           bias['dItems'][j] + \
                           bias['dUsers'][i] + \
                           np.dot(P[i, :], Q[j, :])

                eij = rij_pred - data.Y[k]

                tempP = alpha * 2 * (eij * Q[j] + beta * LA.norm(P[i]))
                tempQ = alpha * 2 * (eij * P[i] + beta * LA.norm(Q[j]))
                P[i] -= tempP
                Q[j] -= tempQ

        # Compute error (this section can be commented)
        if verbose:
            error = 0
            for k in range(0, len(data.Y)):
                i, j = data.X[k]
                rij_pred = self.global_average + \
                           bias['dItems'][j] + \
                           bias['dUsers'][i] + \
                           np.dot(P[i, :], Q[j, :])

                error += (rij_pred - data.Y[k])**2

            error = math.sqrt(error/len(data.Y))
            print('- RMSE: %.3f' % error)

        return P, Q, bias


    def compute_bias(self, data, verbose=False):
        """ Compute averages and biases of the matrix R

        Args:
            data: Orange.data.Table

            verbose: boolean, optional
                If true, it outputs information about the process.

        Returns:
            bias (dictionary: {'delta items' , 'delta users'})

        """

        # Count non zeros in rows and columns
        countings_users = np.bincount(data.X[:, 0])
        countings_items = np.bincount(data.X[:, 1])

        # Sum values along axis 0 and 1
        sums_users = np.bincount(data.X[:, 0], weights=data.Y)
        sums_items = np.bincount(data.X[:, 1], weights=data.Y)

        # Compute averages
        averages_users = sums_users / countings_users
        averages_items = sums_items / countings_items

        # Compute bias and deltas
        deltaUser = averages_users - self.global_average
        deltaItem = averages_items - self.global_average

        bias = {'dItems': deltaItem,
                'dUsers': deltaUser}

        return bias



class BRISMFModel(Model):

    def __init__(self, P, Q, bias, global_average):
        """This model receives a learner and provides and interface to make the
        predictions for a given user.

        Args:
            P: Matrix (users x Latent_factors)

            Q: Matrix (items x Latent_factors)

            bias: dictionary
                'delta items', 'delta users', 'global mean items' and
                'global mean users'

            global_average: float

       """
        self.P = P
        self.Q = Q
        self.bias = bias
        self.global_average = global_average
        self.shape = (len(self.P), len(self.Q))


    def predict(self, X):
        """This function receives an array of indexes [[idx_user, idx_item]] and
        returns the prediction for these pairs.

            Args:
                X: Matrix (2xN)
                    Matrix that contains pairs of the type user-item

            Returns:
                Array with the recommendations for a given user.

            """

        bias = self.global_average + \
               self.bias['dUsers'][X[:, 0]] + \
               self.bias['dItems'][X[:, 1]]

        tempP = self.P[X[:, 0]]
        tempQ = self.Q[X[:, 1]]

        # base_pred = np.multiply(tempP, tempQ)
        base_pred = np.einsum('ij,ij->i', tempP, tempQ)
        predictions = bias + base_pred

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
            users = np.asarray(range(0, len(self.bias['dUsers'])))

        bias = self.global_average + self.bias['dUsers'][users]
        tempB = np.tile(np.array(self.bias['dItems']), (len(users), 1))
        bias = bias[:, np.newaxis] + tempB

        base_pred = np.dot(self.P[users], self.Q.T)
        predictions = bias + base_pred

        # Return top-k recommendations
        if top is not None:
            return predictions[:, :top]

        return predictions


    def __str__(self):
        return self.name