from Orange.data import Table, Domain, ContinuousVariable, StringVariable

from orangecontrib.recommendation import Learner, Model
from orangecontrib.recommendation.utils import format_data

import numpy as np
import math

import time
import warnings

__all__ = ['SVDPlusPlusLearner']

class SVDPlusPlusLearner(Learner):
    """ SVD++: Matrix factorization that also takes into account what users have
    rated

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

    name = 'SVD++'

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
        self.Y = None
        self.bias = None
        self.global_average = None
        self.verbose = verbose
        self.shape = None
        self.order = None
        self.feedback = None

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

        users = np.unique(data.X[:, self.order[0]])
        self.feedback = {}

        for u in users:
            indices_items = np.where(data.X[:, self.order[0]] == u)
            items = data.X[:, self.order[1]][indices_items]
            self.feedback[u] = items

        # Factorize matrix
        self.P, self.Q, self.Y, self.bias = self.matrix_factorization(data,
                                                                  self.feedback,
                                                                  self.K,
                                                                  self.steps,
                                                                  self.alpha,
                                                                  self.beta,
                                                                  self.verbose)

        return SVDPlusPlusModel(P=self.P,
                                Q=self.Q,
                                Y=self.Y,
                                bias=self.bias,
                                global_average=self.global_average,
                                order=self.order,
                                feedback=self.feedback)




    def matrix_factorization(self, data, feedback, K, steps, alpha, beta, verbose=False):
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
        Y = np.random.randn(num_items, K)

        # TODO: missing parameters w_ij and c_ij as in article
        # ...

        # Compute biases
        bias = self.compute_bias(data)

        # Factorize matrix using SGD
        for step in range(steps):
            if verbose:
                start = time.time()
                print('- Step: %d' % (step + 1))

            # Compute predictions
            for k in range(0, len(data.Y)):
                i = data.X[k][self.order[0]]  # User
                j = data.X[k][self.order[1]]  # Item
                f = feedback[i]  # Implicit data

                b_ui = self.global_average + \
                           bias['dItems'][j] + \
                           bias['dUsers'][i]

                norm_denominator = math.sqrt(len(f))
                tempN = np.sum(Y[f], axis=0)
                p_plus_y_sum_vector = tempN/norm_denominator + P[i, :]

                rij_pred = b_ui + np.dot(p_plus_y_sum_vector, Q[j, :])
                eij = rij_pred - data.Y[k]

                tempP = alpha * 2 * (eij * Q[j] + beta * P[i])
                tempQ = alpha * 2 * (eij * p_plus_y_sum_vector + beta * Q[j])
                tempY = alpha * 2 * (eij/norm_denominator * Q[j] + beta * Y[f])

                Q[j] -= tempQ
                P[i] -= tempP
                Y[f] -= tempY


            if verbose:
                print('\tTime: %.3fs' % (time.time() - start))
                print('\tRMSE: %.3f\n' % self.compute_rmse(data, feedback,
                                                           bias, P, Q, Y))

        return P, Q, Y, bias


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


    def compute_rmse(self, data, feedback, bias, P, Q, Y):
        sq_error = 0
        for k in range(0, len(data.Y)):
            i = data.X[k][self.order[0]]  # User
            j = data.X[k][self.order[1]]  # Item
            f = feedback[i]  # Implicit data

            b_ui = self.global_average + \
                   bias['dItems'][j] + \
                   bias['dUsers'][i]

            norm_denominator = math.sqrt(len(f))
            tempN = np.sum(Y[f], axis=0)
            p_plus_y_sum_vector = tempN/norm_denominator + P[i, :]

            rij_pred = b_ui + np.dot(p_plus_y_sum_vector, Q[j, :])

            sq_error += (rij_pred - data.Y[k]) ** 2

        # Compute RMSE
        rmse = math.sqrt(sq_error / len(data.Y))
        return rmse

    def compute_objective(self, data, feedback, bias, P, Q, Y):
        objective = 0
        for k in range(0, len(data.Y)):
            i = data.X[k][self.order[0]]  # User
            j = data.X[k][self.order[1]]  # Item
            f = feedback[i]  # Implicit data

            # Prediction
            b_ui = self.global_average + \
                   bias['dItems'][j] + \
                   bias['dUsers'][i]

            norm_denominator = math.sqrt(len(f))
            tempN = np.sum(Y[f], axis=0)
            p_plus_y_sum_vector = tempN / norm_denominator + P[i, :]

            rij_pred = b_ui + np.dot(p_plus_y_sum_vector, Q[j, :])

            objective += (rij_pred - data.Y[k]) ** 2

            # Regularization
            #TODO: missing parameters w_ij and c_ij as in article to be added in objective function
            objective += self.beta * (np.linalg.norm(P[i, :]) ** 2
                                      + np.linalg.norm(Q[j, :]) ** 2
                                      + bias['dItems'][j] ** 2
                                      + bias['dUsers'][i] ** 2)

        # Compute RMSE
        rmse = math.sqrt(objective / len(data.Y))
        return rmse


class SVDPlusPlusModel(Model):

    def __init__(self, P, Q, Y, bias, global_average, order, feedback):
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
        self.Y = Y
        self.bias = bias
        self.global_average = global_average
        self.shape = (len(self.P), len(self.Q))
        self.order = order
        self.feedback = feedback


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

        base_pred = []
        for k in range(len(X)):
            i = X[k, self.order[0]]
            f = self.feedback[i]  # Implicit data

            norm_denominator = np.sqrt(len(f))
            tempN = np.sum(self.Y[f], axis=0)

            p_plus_y_sum_vector = tempN/norm_denominator + tempP[k]
            base_pred.append(np.dot(p_plus_y_sum_vector, tempQ[k]))

        predictions = bias + np.asarray(base_pred)
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

        predictions = []
        for i in range(len(users)):
            u = users[i]
            f = self.feedback[u]  # Implicit data

            norm_denominator = np.sqrt(len(f))
            tempN = np.sum(self.Y[f], axis=0)

            p_plus_y_sum_vector = tempN / norm_denominator + self.P[u]
            pred = bias[i, :]
            pred2 = np.dot(p_plus_y_sum_vector, self.Q.T)

            predictions.append(pred + pred2)

        predictions = np.asarray(predictions)

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


    def getYTable(self):
        latentFactors_Y = [ContinuousVariable('K' + str(i + 1))
                           for i in range(len(self.Y[0]))]

        variable = self.original_domain.variables[self.order[1]]

        if isinstance(variable, ContinuousVariable):
            domain_val = ContinuousVariable(variable.name)
            values = np.atleast_2d(np.arange(0, len(self.Y))).T
        else:
            domain_val = StringVariable(variable.name)
            values = np.column_stack((variable.values,))

        domain_Y = Domain(latentFactors_Y, None, [domain_val])
        return Table(domain_Y, self.Y, None, values)


    def __str__(self):
        return self.name
