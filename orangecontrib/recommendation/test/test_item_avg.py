import unittest

import numpy as np
from orangecontrib.recommendation import ItemAvgLearner
from sklearn.metrics import mean_squared_error
import math
import Orange

class TestItemAvg(unittest.TestCase):

    def test_ItemAvg_correctness(self):
        filename = '../datasets/users-movies-toy2.tab'
        data = Orange.data.Table(filename)

        # Train recommender
        learner = ItemAvgLearner(verbose=False)
        recommender = learner(data)

        print('Items average: %s' % np.array_str(recommender.items_average))
        ground_truth = np.asarray([3.2, 3.6, 2.7142, 3.2, 3.3333])
        #print(np.mean(ground_truth))
        np.testing.assert_array_almost_equal(recommender.items_average,
                                             ground_truth,
                                             decimal=2)


    def test_ItemAvg_predict_items(self):
        # Load data
        filename = '../datasets/users-movies-toy2.tab'
        data = Orange.data.Table(filename)

        # Train recommender
        learner = ItemAvgLearner(verbose=False)
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


    def test_ItemAvg_input_data(self):
        # Load data
        filename = '../datasets/users-movies-toy2.tab'
        data = Orange.data.Table(filename)

        # Train recommender
        learner = ItemAvgLearner(verbose=False)
        recommender = learner(data)

        # Compute predictions
        y_pred = recommender(data)

        # Compute RMSE
        rmse = math.sqrt(mean_squared_error(data.Y, y_pred))
        print('-> RMSE (input data): %.3f' % rmse)

        # Check correctness
        self.assertGreaterEqual(rmse, 0)


    def test_ItemAvg_pairs(self):
        # Load data
        filename = '../datasets/users-movies-toy2.tab'
        data = Orange.data.Table(filename)

        # Train recommender
        learner = ItemAvgLearner(verbose=False)
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


    def test_ItemAvg_CV(self):
        from Orange.evaluation.testing import CrossValidation

        # Load data
        filename = '../datasets/users-movies-toy2.tab'
        data = Orange.data.Table(filename)

        items_avg = ItemAvgLearner(verbose=False)
        learners = [items_avg]


        # learner = ItemAvgLearner(verbose=False)
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

