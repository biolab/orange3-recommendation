import Orange
from orangecontrib.recommendation import TrustSVDLearner

from sklearn.metrics import mean_squared_error

import unittest
import numpy as np
import math
import random


class TestTrustSVD(unittest.TestCase):

    def test_TrustSVD_predict_items(self):
        ratings = Orange.data.Table('filmtrust/ratings_small.tab')
        trust = Orange.data.Table('filmtrust/trust_small.tab')

        # Train recommender
        learner = TrustSVDLearner(num_factors=2, num_iter=1, learning_rate=0.07,
                                  lmbda=0.1, social_lmbda=0.05, trust=trust,
                                  verbose=False)
        recommender = learner(ratings)

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

    def test_TrustSVD_input_data_continuous(self):
        # Load data
        ratings = Orange.data.Table('filmtrust/ratings_small.tab')
        trust = Orange.data.Table('filmtrust/trust_small.tab')

        # Train recommender
        learner = TrustSVDLearner(num_factors=2, num_iter=1, learning_rate=0.07,
                                  lmbda=0.1, social_lmbda=0.05, trust=trust)
        recommender = learner(ratings)

        print(str(recommender) + ' trained')

        # Compute predictions
        y_pred = recommender(ratings)

        # Compute RMSE
        rmse = math.sqrt(mean_squared_error(ratings.Y, y_pred))
        print('-> RMSE (input data; continuous): %.3f' % rmse)

        # Check correctness
        self.assertGreaterEqual(rmse, 0)

        # Check tables P, Q, Y and W
        P = recommender.getPTable()
        Q = recommender.getQTable()
        Y = recommender.getYTable()
        W = recommender.getWTable()

        diff = len(set([P.X.shape[1], Q.X.shape[1], Y.X.shape[1], W.X.shape[1]]))
        self.assertEqual(diff, 1)


    def test_TrustSVD_CV(self):
        from Orange.evaluation.testing import CrossValidation

        # Load data
        ratings = Orange.data.Table('filmtrust/ratings_small.tab')
        trust = Orange.data.Table('filmtrust/trust_small.tab')

        trustsvd = TrustSVDLearner(num_factors=2, num_iter=1,
                                   learning_rate=0.07, lmbda=0.1,
                                   social_lmbda=0.05, trust=trust)
        learners = [trustsvd]

        res = CrossValidation(ratings, learners, k=3)
        rmse = Orange.evaluation.RMSE(res)
        r2 = Orange.evaluation.R2(res)

        print("Learner  RMSE  R2")
        for i in range(len(learners)):
            print(
                "{:8s} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], r2[i]))

        self.assertIsInstance(rmse, np.ndarray)

    def test_TrustSVD_warnings(self):
        # Load data
        data = Orange.data.Table('filmtrust/ratings_small.tab')
        trust = Orange.data.Table('filmtrust/trust_small.tab')

        # Train recommender
        learner = TrustSVDLearner(num_factors=15, num_iter=1, learning_rate=0.0,
                                  lmbda=0.1, social_lmbda=0.05, trust=trust)

        self.assertWarns(UserWarning, learner, data)

    # def test_TrustSVD_objective(self):
    #     pass
    #     # Load data
    #     # ratings = Orange.data.Table('filmtrust/ratings.tab')
    #     # trust = Orange.data.Table('filmtrust/trust.tab')
    #     #
    #     # steps = [1, 10, 30]
    #     # objectives = []
    #     #
    #     # for step in steps:
    #     #     learner = TrustSVDLearner(num_factors=15, num_iter=step, learning_rate=0.07, lmbda=0.1,
    #     #                               social_lmbda=0.05, trust=trust,
    #     #                               verbose=False)
    #     #     recommender = learner(ratings)
    #     #     objectives.append(
    #     #         recommender.compute_objective(data=ratings, lmbda=learner.beta,
    #     #                                       social_lmbda=learner.beta_t))
    #     #
    #     # # Assert objective values decrease
    #     # test = list(
    #     #     map(lambda t: t[0] >= t[1], zip(objectives, objectives[1:])))
    #     # self.assertTrue(all(test))


if __name__ == "__main__":
    # Test all
    unittest.main()

    # # Test single test
    # suite = unittest.TestSuite()
    # suite.addTest(TestTrustSVD("test_TrustSVD_input_data_continuous"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

