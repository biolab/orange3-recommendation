import unittest

import numpy as np
from orangecontrib.recommendation import BRISMFLearner
from sklearn.metrics import mean_squared_error
import math
import Orange

class TestBRISMF(unittest.TestCase):

    """
    def test_BRISMF_predict_items(self):
        # Load data
        filename = '../datasets/users-movies-toy2.tab'
        data = Orange.data.Table(filename)

        # Train recommender
        learner = BRISMFLearner(K=10, steps=50, verbose=False)
        recommender = learner(data)

        # Compute predictions
        prediction = recommender.predict_items()
        y_pred = prediction[data.X[:, 0], data.X[:, 1]]

        # Compute RMSE
        rmse = math.sqrt(mean_squared_error(data.Y, y_pred))
        print('-> RMSE (predict items): %.3f' % rmse)

        # Check correctness
        self.assertLessEqual(rmse, 1)


    def test_BRISMF_input_data(self):
        # Load data
        filename = '../datasets/users-movies-toy2.tab'
        data = Orange.data.Table(filename)

        # Train recommender
        learner = BRISMFLearner(K=10, steps=50, verbose=False)
        recommender = learner(data)

        # Compute predictions
        y_pred = recommender(data)

        # Compute RMSE
        rmse = math.sqrt(mean_squared_error(data.Y, y_pred))
        print('-> RMSE (input data): %.3f' % rmse)

        # Check correctness
        self.assertLessEqual(rmse, 1)


    def test_BRISMF_pairs(self):
        # Load data
        filename = '../datasets/users-movies-toy2.tab'
        data = Orange.data.Table(filename)

        # Train recommender
        learner = BRISMFLearner(K=10, steps=50, verbose=False)
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
    """

    def test_BRISMF_CV(self):
        from Orange.evaluation.testing import CrossValidation

        # Load data
        filename = '../datasets/users-movies-toy2.tab'
        data = Orange.data.Table(filename)

        brismf = BRISMFLearner(K=15, steps=250, verbose=False)
        learners = [brismf]

        # learner = BRISMFLearner(K=15, steps=250, verbose=False)
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


    """
    def test_normalCV(self):
        data = Orange.data.Table("housing.tab")

        lin = Orange.regression.linear.LinearRegressionLearner()

        learners = [lin]

        res = Orange.evaluation.CrossValidation(data, learners, k=5)
        rmse = Orange.evaluation.RMSE(res)
        r2 = Orange.evaluation.R2(res)

        print("Learner  RMSE  R2")
        for i in range(len(learners)):
            print(
                "{:8s} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], r2[i]))
    """

if __name__ == "__main__":
    unittest.main()

