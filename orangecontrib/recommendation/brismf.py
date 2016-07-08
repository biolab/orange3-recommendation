from Orange.data import Table, Domain, ContinuousVariable, StringVariable

from orangecontrib.recommendation import Learner, Model
from orangecontrib.recommendation.utils import format_data

import numpy as np
import math

import time
import warnings

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
                 K=5,
                 steps=25,
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
                          "well.", stacklevel=2)

        # Optional, can be manage through preprocessors.
        data, self.order, self.shape = format_data.preprocess(data)

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
                           global_average=self.global_average,
                           order=self.order)




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
        P = np.random.rand(num_users, K)  # User and features
        Q = np.random.rand(num_items, K)  # Item and features

        # Compute biases
        bias = self.compute_bias(data)

        # Factorize matrix using SGD
        for step in range(steps):
            if verbose:
                start = time.time()
                print('- Step: %d' % (step + 1))

            # Compute predictions
            for k in range(0, len(data.Y)):
                i = data.X[k][self.order[0]]  # Users
                j = data.X[k][self.order[1]]  # Items

                rij_pred = self.global_average + \
                           bias['dItems'][j] + \
                           bias['dUsers'][i] + \
                           np.dot(P[i], Q[j])

                eij = rij_pred - data.Y[k]

                tempP = alpha * 2 * (eij * Q[j] + beta * P[i])
                tempQ = alpha * 2 * (eij * P[i] + beta * Q[j])
                P[i] -= tempP
                Q[j] -= tempQ

            if verbose:
                print('\tTime: %.3fs' % (time.time() - start))
                print('\tRMSE: %.3f\n' % self.compute_rmse(data,
                                                         bias, P, Q))

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
        # Bincount() returns an array of length np.amax(x)+1. Therefore, items
        # not rated will have a count=0. To avoid division by zero, replace
        # zeros by ones
        countings_users = np.bincount(data.X[:, self.order[0]])
        countings_items = np.bincount(data.X[:, self.order[1]])

        # Replace zeros by ones (Avoid problems of division by zero)
        # This only should happen during Cross-Validation
        countings_users[countings_users == 0] = 1
        countings_items[countings_items == 0] = 1

        # Sum values along axis 0 and 1
        sums_users = np.bincount(data.X[:, self.order[0]], weights=data.Y)
        sums_items = np.bincount(data.X[:, self.order[1]], weights=data.Y)

        # Compute averages
        averages_users = sums_users / countings_users
        averages_items = sums_items / countings_items

        # Compute bias and deltas
        deltaUser = averages_users - self.global_average
        deltaItem = averages_items - self.global_average

        bias = {'dItems': deltaItem,
                'dUsers': deltaUser}

        return bias


    def compute_rmse(self, data, bias, P, Q):
        sq_error = 0
        for k in range(0, len(data.Y)):
            i = data.X[k][self.order[0]]  # Users
            j = data.X[k][self.order[1]]  # Items

            rij_pred = self.global_average + \
                       bias['dItems'][j] + \
                       bias['dUsers'][i] + \
                       np.dot(P[i, :], Q[j, :])
            sq_error += (rij_pred - data.Y[k]) ** 2

        # Compute RMSE
        rmse = math.sqrt(sq_error / len(data.Y))
        return rmse


class BRISMFModel(Model):

    def __init__(self, P, Q, bias, global_average, order):
        """This model receives a learner and provides and interface to make the
        predictions for a given user.

        Args:
            P: Matrix (users x Latent_factors)

            Q: Matrix (items x Latent_factors)

            bias: dictionary
                {'delta items', 'delta users'}

            global_average: float

            order: (int, int)
                Tuple with the index of the columns users and items in X.
                (idx_user, idx_item)

       """
        self.P = P
        self.Q = Q
        self.bias = bias
        self.global_average = global_average
        self.shape = (len(self.P), len(self.Q))
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


        bias = self.global_average + \
               self.bias['dUsers'][X[:, self.order[0]]] + \
               self.bias['dItems'][X[:, self.order[1]]]

        tempP = self.P[X[:, self.order[0]]]
        tempQ = self.Q[X[:, self.order[1]]]

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
            predictions = predictions[:, :top]

        return predictions


    def getPTable(self):
        latentFactors_P = [ContinuousVariable('K' + str(i + 1))
                           for i in range(len(self.P[0]))]

        variable = self.original_domain.variables[self.order[0]]

        if isinstance(variable, ContinuousVariable):
            domain_val = ContinuousVariable(variable.name)
            values = np.atleast_2d(np.arange(0, len(self.P))).T
        else:
            domain_val = StringVariable(variable.name)
            values = np.column_stack((variable.values,))

        domain_P = Domain(latentFactors_P, None, [domain_val])
        return Table(domain_P, self.P, None, values)


    def getQTable(self):
        latentFactors_Q = [ContinuousVariable('K' + str(i + 1))
                           for i in range(len(self.Q[0]))]

        variable = self.original_domain.variables[self.order[1]]

        if isinstance(variable, ContinuousVariable):
            domain_val = ContinuousVariable(variable.name)
            values = np.atleast_2d(np.arange(0, len(self.Q))).T
        else:
            domain_val = StringVariable(variable.name)
            values = np.column_stack((variable.values,))

        domain_Q = Domain(latentFactors_Q, None, [domain_val])
        return Table(domain_Q, self.Q, None, values)


    def __str__(self):
        return self.name
