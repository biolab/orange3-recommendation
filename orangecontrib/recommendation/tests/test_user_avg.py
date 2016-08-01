import Orange
from orangecontrib.recommendation import UserAvgLearner

from sklearn.metrics import mean_squared_error

import unittest
import numpy as np
import math
import random


class TestUserAvg(unittest.TestCase):

    def test_UserAvg_correctness(self):

        # Load data
        data = Orange.data.Table('ratings.tab')

        # Train recommender
        learner = UserAvgLearner(verbose=False)
        recommender = learner(data)

        # Set ground truth
        ground_truth = np.asarray([3.25, 3.6666, 3.25, 3.3333, 3.3333, 2.6666,
                                   2.6, 3.6666])

        # Compare results
        users_avg = recommender.bias['globalAvg'] + recommender.bias['dUsers']
        np.testing.assert_array_almost_equal(users_avg, ground_truth, decimal=2)

    def test_UserAvg_predict_items(self):

        # Load data
        data = Orange.data.Table('ratings.tab')

        # Train recommender
        learner = UserAvgLearner(verbose=True)
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

    def test_UserAvg_input_data(self):

        # Load data
        print(Orange.data.table.dataset_dirs)
        data = Orange.data.Table('ratings.tab')

        # Train recommender
        learner = UserAvgLearner(verbose=False)
        recommender = learner(data)
        print(str(recommender) + ' trained')

        # Compute predictions
        y_pred = recommender(data)
        y_pred2 = recommender(data[:, recommender.order[0]])

        # Compute RMSE
        rmse = math.sqrt(mean_squared_error(data.Y, y_pred))
        print('-> RMSE (input data): %.3f' % rmse)

        # Check correctness
        self.assertGreaterEqual(rmse, 0)
        np.testing.assert_equal(y_pred, y_pred2)

    def test_UserAvg_pairs(self):

        # Load data
        data = Orange.data.Table('ratings.tab')

        # Train recommender
        learner = UserAvgLearner(verbose=False)
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

    def test_UserAvg_CV(self):
        from Orange.evaluation.testing import CrossValidation

        # Load data
        data = Orange.data.Table('ratings.tab')

        users_avg = UserAvgLearner(verbose=False)
        learners = [users_avg]

        res = CrossValidation(data, learners, k=3)
        rmse = Orange.evaluation.RMSE(res)
        r2 = Orange.evaluation.R2(res)

        print("Learner  RMSE  R2")
        for i in range(len(learners)):
            print(
                "{:8s} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], r2[i]))

        self.assertIsInstance(rmse, np.ndarray)


if __name__ == "__main__":
    #unittest.main()

    # Test single test
    suite = unittest.TestSuite()
    suite.addTest(TestUserAvg("test_UserAvg_input_data"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

