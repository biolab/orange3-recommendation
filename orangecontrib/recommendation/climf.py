from Orange.data import Table, Domain, ContinuousVariable, StringVariable

from orangecontrib.recommendation import Learner, Model
from orangecontrib.recommendation.utils import format_data
from orangecontrib.recommendation.evaluation import MeanReciprocalRank

import numpy as np
from scipy.special import expit as sigmoid
from scipy.sparse import dok_matrix

import time
import random
import warnings

__all__ = ['CLiMFLearner']


def g(x):
    """sigmoid function"""
    return sigmoid(x)


def dg(x):
    ex = np.exp(-x)
    y = ex / (1 + ex) ** 2
    return y


class CLiMFLearner(Learner):
    """ Collaborative Less-is-More Filtering Matrix Factorization

    Matrix factorization for scenarios with binary relevance data when only a
    few (k) items are recommended to individual users. It improves top-k
    recommendations through ranking by directly maximizing the Mean Reciprocal
    Rank (MRR).


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
                          "well.", stacklevel=2)

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

                    dV = g(-f[j]) - beta * V[w]

                    # For I
                    vec1 = dg(f[j] - f) * \
                        (1/(1 - g(f - f[j])) - 1/(1 - g(f[j] - f)))
                    dV += np.einsum('i,j->ij', vec1, U[i]).sum(axis=0)

                    V[w] += alpha * dV
                    dU += g(-f[j]) * V[w]

                    # For II
                    vec2 = (V[items[j]] - V[items])
                    vec3 = dg(f - f[j])/(1 - g(f - f[j]))
                    dU += np.einsum('ij,i->ij', vec2, vec3).sum(axis=0)

                U[i] += alpha * dU

            if verbose:
                print('\tTime: %.3fs' % (time.time() - start))
                print('\tMRR = {0:.4f}\n'.format(
                    self.compute_mrr(data.X, U,V, train_sample_users)))

        return U, V


    def compute_mrr(self, X, U, V, test_users):
        start = time.time()

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

        print('\tTime MRR: %.3fs' % (time.time() - start))
        return MRR


    def compute_objective(self, X, Y, U, V):

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
                W2[i, j] = sum((np.log(1 - Ys[i, k] * sigmoid(U[i, :].dot(V[k, :] - V[j, :])))
                                       for k in range(N)))

        objective = Ys.multiply(W1 - W2).sum()
        objective -= self.beta/2.0 * (np.linalg.norm(U, ord="fro")**2
                                       + np.linalg.norm(V, ord="fro") ** 2)

        return objective


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
            predictions = predictions[:, :top_k]

        return predictions



    def getUTable(self):
        latentFactors_U = [ContinuousVariable('K' + str(i + 1))
                           for i in range(len(self.U[0]))]

        variable = self.original_domain.variables[self.order[0]]

        if isinstance(variable, ContinuousVariable):
            domain_val = ContinuousVariable(variable.name)
            values = np.atleast_2d(np.arange(0, len(self.U))).T
        else:
            domain_val = StringVariable(variable.name)
            values = np.column_stack((variable.values,))

        domain_U = Domain(latentFactors_U, None, [domain_val])
        return Table(domain_U, self.U, None, values)


    def getVTable(self):
        latentFactors_V = [ContinuousVariable('K' + str(i + 1))
                           for i in range(len(self.V[0]))]

        variable = self.original_domain.variables[self.order[1]]

        if isinstance(variable, ContinuousVariable):
            domain_val = ContinuousVariable(variable.name)
            values = np.atleast_2d(np.arange(0, len(self.V))).T
        else:
            domain_val = StringVariable(variable.name)
            values = np.column_stack((variable.values,))

        domain_V = Domain(latentFactors_V, None, [domain_val])
        return Table(domain_V, self.V, None, values)


    def __str__(self):
        return self.name

