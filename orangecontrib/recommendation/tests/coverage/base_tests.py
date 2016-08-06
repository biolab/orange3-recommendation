import Orange

from sklearn.metrics import mean_squared_error

import unittest
import numpy as np
import math
import random

__all__ = ["TestRatingModels"]


class TestRatingModels:

    # @classmethod
    # def setUpClass(cls):
    #     if cls is TestRatingModels:
    #         raise unittest.SkipTest("Skip BaseTest tests, it's a base class")
    #     super(TestRatingModels, cls).setUpClass()

    def test_input_data_discrete(self, learner, filename):
        # Load data
        data = Orange.data.Table(filename)

        # Train recommender
        recommender = learner(data)
        print(str(recommender) + ' trained')

        # Compute predictions
        y_pred = recommender(data)

        # Compute RMSE
        rmse = math.sqrt(mean_squared_error(data.Y, y_pred))
        print('-> RMSE (input data; discrete): %.3f' % rmse)

        # Check correctness
        self.assertGreaterEqual(rmse, 0)

    def test_input_data_continuous(self, learner, filename):
        # Load data
        data = Orange.data.Table(filename)

        # Train recommender
        recommender = learner(data)

        print(str(recommender) + ' trained')

        # Compute predictions
        y_pred = recommender(data)

        # Compute RMSE
        rmse = math.sqrt(mean_squared_error(data.Y, y_pred))
        print('-> RMSE (input data; continuous): %.3f' % rmse)

        # Check correctness
        self.assertGreaterEqual(rmse, 0)

    def test_pairs(self, learner, filename):
        # Load data
        data = Orange.data.Table(filename)

        # Train recommender
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

    def test_predict_items(self, learner, filename):
        # Load data
        data = Orange.data.Table(filename)

        # Train recommender
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

    def test_CV(self, learner, filename):
        from Orange.evaluation.testing import CrossValidation

        # Load data
        data = Orange.data.Table(filename)

        learners = [learner]

        res = CrossValidation(data, learners, k=3)
        rmse = Orange.evaluation.RMSE(res)
        r2 = Orange.evaluation.R2(res)

        print("Learner  RMSE  R2")
        for i in range(len(learners)):
            print(
                "{:8s} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], r2[i]))

        self.assertIsInstance(rmse, np.ndarray)

    def test_warnings(self, learner, filename):
        # Load data
        data = Orange.data.Table(filename)

        # Train recommender and check warns
        self.assertWarns(UserWarning, learner, data)

if __name__ == "__main__":
    # Test all
    unittest.main()
