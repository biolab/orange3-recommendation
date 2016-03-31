from Orange.base import Model, Learner

import numpy as np
from numpy import linalg as LA
from scipy import sparse

import warnings

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

        # Factorize matrix
        self.P, self.Q, self.bias = self.matrix_factorization(
                                                    X,
                                                    self.K,
                                                    self.steps,
                                                    self.alpha,
                                                    self.beta)

        return BRISMFModel(self)

    def matrix_factorization(self, R, K, steps, alpha, beta):

        # Initialize factorized matrices randomly
        num_users, num_items = R.shape
        P = np.random.rand(num_users, K)  # User and features
        Q = np.random.rand(num_items, K)  # Item and features

        # Local means (array)
        mean_user_rating = R.mean(axis=1)  # Rows
        mean_item_rating = R.mean(axis=0)  # Columns

        # Global mean
        global_mean_users = mean_user_rating.mean()
        global_mean_items = mean_item_rating.mean()

        # Compute bias and deltas
        deltaUser = mean_user_rating - global_mean_users
        deltaItem = mean_item_rating - global_mean_items
        #bias_items = np.full((num_users, num_items), global_mean_items)
        #bias = ((bias_items + deltaItem).T + deltaUser).T

        # Get non-zero elements
        indices = np.array(R.nonzero()).T

        # Factorize matrix using SGD
        for step in range(steps):

             # Compute predictions
            for i, j in indices:
                rij_pred = global_mean_items + \
                           deltaItem[0, j] + \
                           deltaUser[i, 0] + \
                           np.dot(P[i, :], Q[j, :])

                eij = R[i, j] - rij_pred
                for k in range(K):
                    P[i][k] += alpha * (2 * eij * Q[j][k] - beta * P[i][k])
                    Q[j][k] += alpha * (2 * eij * P[i][k] - beta * Q[j][k])


            # Compute error
            e = 0
            for i, j in indices:
                if R[i, j] > 0:
                    rij_pred = global_mean_items + \
                               deltaItem[0, j] + \
                               deltaUser[i, 0] + \
                               np.dot(P[i, :], Q[j, :])

                    e += pow(R[i, j] - rij_pred, 2)
                    for k in range(K):
                        e += (beta/2) * (pow(P[i][k], 2) + pow(Q[j][k], 2))

            #print(e)
            if e < 0.001:
                print('Converged')
                break
        print('ERROR: ' + str(e))

        #nR = np.dot(P, Q) + bias
        #print(nR)
        bias = [[global_mean_users, global_mean_items], [deltaUser, deltaItem]]
        return P, Q, bias


class BRISMFModel(Model):

    def predict(self, user, sort=True, top=None):
        """ Sort recomendations for user """
        global_mean_users, global_mean_items = self.domain.bias[0]
        deltaUser, deltaItem  = self.domain.bias[1]


        #bias_items = np.full((num_users, num_items), global_mean_items)
        #bias = ((bias_items + deltaItem).T + deltaUser).T

        bias = global_mean_items + deltaItem + deltaUser[user, 0]
        base_pred = np.dot(self.domain.P[user], self.domain.Q.T)
        predictions = np.array(bias[0, :] + base_pred)[0]


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


    # Transform to sparse matrix
    data_sparse = sparse.csr_matrix(ratings_matrix)
    #data_sparse = ratings_matrix

    import time
    start = time.time()
    learner = BRISMFLearner()
    recommender = learner.fit(X=ratings_matrix, Y=None, W=None)

    prediction = recommender.predict(user=1, sort=False, top=None)
    print('Time: %.3fs\n' % (time.time() - start))

    print(ratings_matrix)
    print('')
    print(prediction[:, 1].T)
    #correct = np.array([4,  1,  3,  1])
    #np.testing.assert_almost_equal(
    #    np.round(np.round(prediction[:, 1])), np.round(correct))



if __name__ == "__main__":
    test_BRISMF()