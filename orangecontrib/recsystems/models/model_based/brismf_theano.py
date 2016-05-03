from Orange.base import Model, Learner

import numpy as np
from numpy import linalg as LA

import theano
import theano.tensor as T
import lasagne

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

        X = self.prepare_data(X)

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

        # Local means (array)
        mean_user_rating = np.mean(R, axis=1)  # Rows
        mean_item_rating = np.mean(R, axis=0)  # Columns

        # Global mean
        global_mean_users = np.mean(mean_user_rating)
        global_mean_items = np.mean(mean_item_rating)

        # Compute bias and deltas
        deltaUser = mean_user_rating - global_mean_users
        deltaItem = mean_item_rating - global_mean_items
        bias_items = np.full((num_users, num_items), global_mean_items)
        bias = ((bias_items + deltaItem).T + deltaUser).T

        # Get non-zero elements
        indices = np.array(np.nonzero(R > 0)).T
        error = 0

        # Factorize matrix using SGD
        for step in range(steps):

            # Compute predictions
            for i, j in indices:
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
            for i, j in indices:
                rij_pred = global_mean_items + \
                           deltaItem[j] + \
                           deltaUser[i] + \
                           np.dot(P[i, :], Q[j, :])
                error += abs(rij_pred - R[i, j])

            error = error / len(indices)
            if error < 0.001:
               print('Converged')
               break
        print('ERROR: ' + str(error))

        return P, Q, bias


class BRISMFModel(Model):

    def predict(self, user, sort=True, top=None):
        """ Sort recomendations for user """
        bias = self.domain.bias[user, :]
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


    import time
    start = time.time()

    learner = BRISMFLearner()
    recommender = learner.fit(X=ratings_matrix, Y=None, W=None)
    print('Time: %.3fs\n' % (time.time() - start))

    print(ratings_matrix)
    print('')
    for i in range(0, 5):
        prediction = recommender.predict(user=i, sort=False, top=None)
        print(prediction[:, 1].T)
    #correct = np.array([4,  1,  3,  1])
    #np.testing.assert_almost_equal(
    #    np.round(np.round(prediction[:, 1])), np.round(correct))



if __name__ == "__main__":
    test_BRISMF()