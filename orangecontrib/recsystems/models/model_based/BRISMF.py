from Orange.base import Model, Learner

import numpy as np
from numpy import linalg as LA

from scipy import sparse

#import theano
#import theano.tensor as T

import math
import random

try:
    from numba import jit
except ImportError:
    jit = lambda x: x

import warnings
import time

__all__ = ['BRISMFLearner']

class BRISMFLearner(Learner):

    def __init__(self,
                 K=2,
                 steps=100,
                 alpha=0.005,
                 beta=0.02,
                 preprocessors=None):
        self.K = K
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.P = None
        self.Q = None
        self.bias = None

        super().__init__(preprocessors=preprocessors)
        self.params = vars()


    def fit(self, X, Y, W):

        # Warnings
        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well. You are advised to use the LinearRegression "
                          "estimator", stacklevel=2)

        #X = self.prepare_data(X)

        # Factorize matrix
        self.P, self.Q, self.bias = self.matrix_factorization(
                                                                X,
                                                                self.K,
                                                                self.steps,
                                                                self.alpha,
                                                                self.beta)

        return BRISMFModel(self)

    def prepare_data(self, X):

        # Convert NaNs to zero
        where_are_NaNs = np.isnan(X)
        X[where_are_NaNs] = 0

        # Transform dense matrix into sparse matrix
        #X = sparse.csr_matrix(X)

        return X


    def matrix_factorization(self, R, K, steps, alpha, beta):

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
            print('- Time mean (sparse): %.3fs\n' % (time.time() - start2))

        else:  # Dense matrix
            start2 = time.time()
            # Local means (array)
            mean_user_rating = np.mean(R, axis=1)  # Rows
            mean_item_rating = np.mean(R, axis=0)  # Columns

            # Global mean
            global_mean_users = np.mean(mean_user_rating)
            global_mean_items = np.mean(mean_item_rating)
            print('- Time mean (dense): %.3fs\n' % (time.time() - start2))


        # Compute bias and deltas (Common - Dense/Sparse matrices)
        deltaUser = mean_user_rating - global_mean_users
        deltaItem = mean_item_rating - global_mean_items
        bias = {'dItems': deltaItem,
                'dUsers': deltaUser,
                'gMeanItems': global_mean_items,
                'gMeanUsers': global_mean_users}


        # Get non-zero elements
        indices = np.array(np.nonzero(R > 0)).T

        # Factorize matrix using SGD
        for step in range(steps):

            # Compute predictions
            for i, j in indices:
                    if R[i, j] > 0:
                        rij_pred = global_mean_items + \
                                   deltaItem[j] + \
                                   deltaUser[i] + \
                                   np.dot(P[i, :], Q[j, :])

                        eij = rij_pred - R[i, j]

                        tempP = alpha * 2 * (eij * Q[j] + beta * LA.norm(P[i]))
                        tempQ = alpha * 2 * (eij * P[i] + beta * LA.norm(Q[j]))
                        P[i] -= tempP
                        Q[j] -= tempQ



        # Compute error
        counter=0
        error=0
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

    # Predict top-best items for a user
    def predict(self, user, sort=True, top=None):
        """ Sort recomendations for user """

        bias = self.domain.bias['gMeanItems'] + \
                    self.domain.bias['dUsers'][user] + \
                    self.domain.bias['dItems']
        base_pred = np.dot(self.domain.P[user], self.domain.Q.T)
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
        return 'BRISMF {}'.format('---> return model')



# Create a random but realistic dataset
def random_dataset(ratings, max_users, max_items, sparse_mat=False):
    MIN_RATING = 1
    MAX_RATING = 5

    ratings_mat = np.random.randint(MIN_RATING, MAX_RATING + 1, ratings)
    users = np.random.randint(0, max_users, ratings)
    items = np.random.randint(0, max_items, ratings)

    # Fill array with zeros
    ratings_matrix = np.zeros((max_users, max_items))

    # Put ratings
    indices = np.column_stack((users, items))
    ratings_matrix[indices[:, 0], indices[:, 1]] += ratings_mat

    if sparse_mat:
        ratings_matrix = sparse.csr_matrix(ratings_matrix)

    return ratings_matrix


def test_BRISMF():
    ratings_matrix = np.array([
        [2, 0, 0, 4, 5, 0],
        [5, 0, 4, 0, 0, 1],
        [0, 0, 5, 0, 2, 0],
        [0, 1, 0, 5, 0, 4],
        [0, 0, 4, 0, 0, 2],
        [4, 5, 0, 1, 0, 0]
    ])




    ratings_matrix = np.asarray([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])

    #MOVIELENS 100K
    # RATINGS = 100000;
    # MAX_USERS = 1000;
    # MAX_ITEMS = 1700;

    # MOVIELENS 1M
    # RATINGS = 1000209;
    # MAX_USERS = 6040;
    # MAX_ITEMS = 3706;

    # NETFLIX
    # RATINGS = 3;
    # MAX_USERS = 480000;
    # MAX_ITEMS = 17000;

    # SIMPLE TEST
    RATINGS = 150;
    MAX_USERS = 100;
    MAX_ITEMS = 25;

    ratings_matrix = random_dataset(ratings=RATINGS,
                                     max_users=MAX_USERS,
                                     max_items=MAX_ITEMS,
                                     sparse_mat=False)

    # Convert to sparse
    #ratings_matrix = sparse.csc_matrix(ratings_matrix)
    print(ratings_matrix)

    start = time.time()

    learner = BRISMFLearner()
    recommender = learner.fit(X=ratings_matrix, Y=None, W=None)

    # print('- Sparsity: %.2f%%\n' % ((RATINGS * 100) / (MAX_USERS * MAX_ITEMS)))
    print('- Time: %.3fs\n' % (time.time() - start))



    """
    print('')
    num_users, num_items = ratings_matrix.shape
    for i in range(0, num_users):
        prediction = recommender.predict(user=i, sort=False, top=None)
        print(prediction[:, 1].T)
    """

    #correct = np.array([4,  1,  3,  1])
    #np.testing.assert_almost_equal(
    #    np.round(np.round(prediction[:, 1])), np.round(correct))



if __name__ == "__main__":
    test_BRISMF()