import os
import math
import unittest

import Orange
from orangecontrib.recommendation import CLiMFLearner
from orangecontrib.recommendation.evaluation import MeanReciprocalRank

import numpy as np
import random


class TestCLiMF(unittest.TestCase):


    def test_CLiMF_input_data(self):
        # Load data
        filename = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../datasets/binary_data.tab'))
        data = Orange.data.Table(filename)

        # Train recommender
        learner = CLiMFLearner(K=10, steps=10, alpha=0.0001, beta=0.001, verbose=True)
        recommender = learner(data)
        print(str(recommender) + ' trained')

        # Select subset to test
        num_sample = min(recommender.shape[0], 100)
        test_users = random.sample(range(recommender.shape[0]),num_sample)

        # Compute predictions
        y_pred = recommender(data[test_users],
                              top_k=min(recommender.shape[1], 5))

        # Compute predictions (Second calling type)
        y_pred2 = recommender(data[test_users].X,
                             top_k=min(recommender.shape[1], 5))

        # Get relevant items for the user
        all_items_u = []
        for i in test_users:
            items_u = data.X[data.X[:, recommender.order[0]] == i][:, recommender.order[1]]
            all_items_u.append(items_u)

        # Compute MRR
        mrr = MeanReciprocalRank(results=y_pred, query=all_items_u)
        print('-> MRR (input data): %.3f' % mrr)

        # Check correctness
        self.assertGreaterEqual(mrr, 0)
        np.testing.assert_equal(y_pred, y_pred2)


    def test_CLiMF_CV(self):
        pass
        # from Orange.evaluation.testing import CrossValidation
        #
        # # Load data
        # filename = '../datasets/binary_data.tab'
        # data = Orange.data.Table(filename)
        #
        # brismf = CLiMFLearner(K=10, steps=10, alpha=0.0001, beta=0.001, verbose=False)
        # learners = [brismf]
        #
        # res = CrossValidation(data, learners, k=5)
        # mrr = MeanReciprocalRank(res)
        #
        # print("Learner  MRR")
        # for i in range(len(learners)):
        #     print(
        #         "{:8s} {:.3f}".format(learners[i].name, mrr[i]))
        #
        # self.assertIsInstance(mrr, np.ndarray)

if __name__ == "__main__":
    # Test all
    #unittest.main()

    # Test single test
    suite = unittest.TestSuite()
    suite.addTest(TestCLiMF("test_CLiMF_input_data"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
