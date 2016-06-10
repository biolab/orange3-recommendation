import unittest

import numpy as np
from orangecontrib.recommendation import UserItemBaselineLearner
from sklearn.metrics import mean_squared_error
import math
import Orange

class TestUserItemBaseline(unittest.TestCase):

    def test_UserItemBaseline_swap_columns(self):
        # Recommender
        learner = UserItemBaselineLearner(verbose=False)

        # Dataset 1
        filename = '../datasets/users-movies-toy.tab'
        data = Orange.data.Table(filename)
        recommender = learner(data)
        prediction = recommender.predict_items()
        y_pred1 = prediction[data.X[:, recommender.order[0]],
                             data.X[:, recommender.order[1]]]

        # Dataset 2
        filename = '../datasets/users-movies-toy2.tab'
        data = Orange.data.Table(filename)
        recommender = learner(data)
        prediction = recommender.predict_items()
        y_pred2 = prediction[data.X[:, recommender.order[0]],
                             data.X[:, recommender.order[1]]]

        # Compare results
        np.testing.assert_array_equal(y_pred1, y_pred2)




    def test_UserAvg_correctness(self):
        filename = '../datasets/users-movies-toy.tab'
        data = Orange.data.Table(filename)

        # Train recommender
        learner = UserItemBaselineLearner(verbose=False)
        recommender = learner(data)

        # Set ground truth
        ground_truth_dItems = [0.02142857, 0.42142857, -0.46428571, 0.02142857,
                               0.1547619]
        ground_truth_dUsers = [0.07142857, 0.48809524, 0.07142857, 0.1547619,
                               0.1547619, -0.51190476, -0.57857143, 0.48809524]

        # Compare results
        np.testing.assert_array_almost_equal(recommender.bias['dItems'],
                                             ground_truth_dItems,
                                             decimal=2)
        np.testing.assert_array_almost_equal(recommender.bias['dUsers'],
                                             ground_truth_dUsers,
                                             decimal=2)
        self.assertAlmostEqual(recommender.global_average, 3.17857, places=2)


    def test_UserItemBaseline_predict_items(self):
        # Load data
        filename = '../datasets/users-movies-toy.tab'
        data = Orange.data.Table(filename)

        # Train recommender
        learner = UserItemBaselineLearner(verbose=False)
        recommender = learner(data)

        # Compute predictions
        prediction = recommender.predict_items()
        y_pred = prediction[data.X[:, recommender.order[0]],
                            data.X[:, recommender.order[1]]]

        # Compute RMSE
        rmse = math.sqrt(mean_squared_error(data.Y, y_pred))
        print('-> RMSE (predict items): %.3f' % rmse)

        # Check correctness
        self.assertGreaterEqual(rmse, 0)


    def test_UserItemBaseline_input_data(self):
        # Load data
        filename = '../datasets/users-movies-toy.tab'
        data = Orange.data.Table(filename)

        # Train recommender
        learner = UserItemBaselineLearner(verbose=False)
        recommender = learner(data)

        # Compute predictions
        y_pred = recommender(data)

        # Compute RMSE
        rmse = math.sqrt(mean_squared_error(data.Y, y_pred))
        print('-> RMSE (input data): %.3f' % rmse)

        # Check correctness
        self.assertGreaterEqual(rmse, 0)


    def test_UserItemBaseline_pairs(self):
        # Load data
        filename = '../datasets/users-movies-toy.tab'
        data = Orange.data.Table(filename)

        # Train recommender
        learner = UserItemBaselineLearner(verbose=False)
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


    def test_UserItemBaseline_CV(self):
        from Orange.evaluation.testing import CrossValidation

        # Load data
        filename = '../datasets/users-movies-toy.tab'
        data = Orange.data.Table(filename)

        user_item_baseline = UserItemBaselineLearner(verbose=False)
        learners = [user_item_baseline]

        # learner = UserItemBaselineLearner(verbose=False)
        # recommender = learner(data)
        # prediction = recommender.predict_items()
        # print(prediction)
        # y_pred = prediction[data.X[:, 0], data.X[:, 1]]
        # rmse = math.sqrt(mean_squared_error(data.Y, y_pred))
        # print('-> RMSE (predict items): %.3f' % rmse)
        # print(recommender.domain.variables[0].values)
        # print(recommender.domain.variables[1].values)
        # print('')

        res = CrossValidation(data, learners, k=5)
        rmse = Orange.evaluation.RMSE(res)
        r2 = Orange.evaluation.R2(res)

        print("Learner  RMSE  R2")
        for i in range(len(learners)):
            print(
                "{:8s} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], r2[i]))

        self.assertIsInstance(rmse, np.ndarray)

if __name__ == "__main__":
    unittest.main()
