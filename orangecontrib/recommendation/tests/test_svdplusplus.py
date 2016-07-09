import os
import math
import random
import unittest

import Orange
from orangecontrib.recommendation import SVDPlusPlusLearner

import numpy as np
from sklearn.metrics import mean_squared_error

class TestSVDPlusPlus(unittest.TestCase):

    def test_SVDPlusPlus_predict_items(self):
        # Load data
        data = Orange.data.Table('ratings3.tab')

        # Train recommender
        learner = SVDPlusPlusLearner(K=2, steps=1, verbose=True)
        recommender = learner(data)

        # Compute predictions 1
        prediction = recommender.predict_items(users=None, top=None)

        # Compute predictions 2 (Execute the other branch)
        num_users = min(recommender.shape[0], 5)
        num_items = min(recommender.shape[1], 5)
        setUsers = random.sample(range(recommender.shape[0]), num_users)
        prediction2 = recommender.predict_items(users=setUsers, top=num_items)

        # Check correctness 1
        len_u, len_i = prediction.shape
        self.assertEqual(len_u, recommender.shape[0])
        self.assertEqual(len_i, recommender.shape[1])

        # Check correctness 2
        len_u, len_i = prediction2.shape
        self.assertEqual(len_u, num_users)
        self.assertEqual(len_i, num_items)

    def test_BRISMF_input_data_discrete(self):
        # Load data
        data = Orange.data.Table('ratings.tab')

        # Train recommender
        learner = SVDPlusPlusLearner(K=2, steps=1, verbose=False)
        recommender = learner(data)
        print(str(recommender) + ' trained')

        # Compute predictions
        y_pred = recommender(data)

        # Compute RMSE
        rmse = math.sqrt(mean_squared_error(data.Y, y_pred))
        print('-> RMSE (input data; discrete): %.3f' % rmse)

        # Check correctness
        self.assertGreaterEqual(rmse, 0)

        # Check tables P and Q
        P = recommender.getPTable()
        Q = recommender.getQTable()
        self.assertEqual(P.X.shape[1], Q.X.shape[1])

    def test_SVDPlusPlus_input_data_continuous(self):
        pass
        # Load data
        data = Orange.data.Table('ratings3.tab')

        # Train recommender
        learner = SVDPlusPlusLearner(K=2, steps=1, verbose=True)
        recommender = learner(data)

        print(str(recommender) + ' trained')

        # Compute predictions
        y_pred = recommender(data)

        # Compute RMSE
        rmse = math.sqrt(mean_squared_error(data.Y, y_pred))
        print('-> RMSE (input data; continuous): %.3f' % rmse)

        # Check correctness
        self.assertGreaterEqual(rmse, 0)

        # # Check tables P and Q
        P = recommender.getPTable()
        Q = recommender.getQTable()
        Y = recommender.getYTable()
        self.assertEqual(P.X.shape[1], Q.X.shape[1], Y.X.shape[1])


    def test_SVDPlusPlus_pairs(self):
        # Load data
        data = Orange.data.Table('ratings3.tab')

        # Train recommender
        learner = SVDPlusPlusLearner(K=2, steps=1, verbose=False)
        recommender = learner(data)

        # Create indices to test
        sample_size = 10
        num_users, num_items = recommender.shape
        idx_users = np.random.randint(0, num_users, size=sample_size)
        idx_items = np.random.randint(0, num_items, size=sample_size)
        indices = np.column_stack((idx_users, idx_items))

        # Compute predictions
        y_pred = recommender(indices)
        print('-> Same number? (pairs): %r' % (len(y_pred) == sample_size))

        # Check correctness
        self.assertEqual(len(y_pred), sample_size)


    def test_SVDPlusPlus_CV(self):
        from Orange.evaluation.testing import CrossValidation

        # Load data
        data = Orange.data.Table('ratings3.tab')

        svdpp = SVDPlusPlusLearner(K=2, steps=1, verbose=False)
        learners = [svdpp]

        res = CrossValidation(data, learners, k=3)
        rmse = Orange.evaluation.RMSE(res)
        r2 = Orange.evaluation.R2(res)

        print("Learner  RMSE  R2")
        for i in range(len(learners)):
            print(
                "{:8s} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], r2[i]))

        self.assertIsInstance(rmse, np.ndarray)


    def test_SVDPlusPlus_warnings(self):

        # Load data
        data = Orange.data.Table('ratings3.tab')

        # Train recommender
        learner = SVDPlusPlusLearner(K=2, steps=1, alpha=0.0, verbose=False)

        self.assertWarns(
            UserWarning,
            learner,
            data
        )

if __name__ == "__main__":
    # Test all
    unittest.main()

    # Test single test
    # suite = unittest.TestSuite()
    # suite.addTest(TestSVDPlusPlus("test_BRISMF_input_data_discrete"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

