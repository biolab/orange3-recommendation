from Orange.base import Model, Learner

import numpy as np
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

        # Factorize matrix using SGD
        for step in range(steps):

            # Compute predictions
            for i, j in indices:
                rij_pred = global_mean_items + \
                           deltaItem[j] + \
                           deltaUser[i] + \
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
                               deltaItem[j] + \
                               deltaUser[i] + \
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

        return P, Q, bias


    def god_factorization(self, R, K, steps, alpha, beta):

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

        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                           target_var)
        loss = loss.mean()

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.001, momentum=0.9)

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()
        # As a bonus, also create an expression for the classification accuracy:
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss,
                                   updates=updates)


        for epoch in range(num_epochs):
            print("Stating epoch {} of {}:".format(epoch + 1, num_epochs))

            train_err = 0
            train_batches = 0

            val_err = 0
            val_acc = 0
            val_batches = 0

            start_time = time.time()
            train_err += train_fn(inputs, targets)

            print("\tEpoch {} took {:.3f}s".format(epoch + 1,
                                                   time.time()-start_time))
            print("\t\t- Training loss:\t\t{:.6f}".format(
                                                    train_err/train_batches))


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

    prediction = recommender.predict(user=4, sort=False, top=None)
    print('Time: %.3fs\n' % (time.time() - start))

    print(ratings_matrix)
    print('')
    print(prediction[:, 1].T)
    #correct = np.array([4,  1,  3,  1])
    #np.testing.assert_almost_equal(
    #    np.round(np.round(prediction[:, 1])), np.round(correct))



if __name__ == "__main__":
    test_BRISMF()