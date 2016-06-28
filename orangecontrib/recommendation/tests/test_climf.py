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
        data = Orange.data.Table('binary_data.tab')

        # Train recommender
        learner = CLiMFLearner(K=2, steps=1, alpha=0.0001, beta=0.001, verbose=True)
        recommender = learner(data)
        print(str(recommender) + ' trained')

        # Create set to test
        num_users = min(recommender.shape[0], 5)
        num_items = recommender.shape[1]
        test_users = random.sample(range(recommender.shape[0]), num_users)

        # Compute predictions 1
        y_pred = recommender(data[test_users], top_k=None)

        # Compute predictions 2 (Execute the 2nd branch)
        y_pred2 = recommender(data[test_users].X, top_k=num_items)

        # Compute predictions 3 (Execute the 3rd branch, "no arg")
        y_pred3 = recommender(data[test_users], no_real_arg='Something')

        # Get relevant items for the user (to test MRR)
        all_items_u = []
        for i in test_users:
            items_u = data.X[data.X[:, recommender.order[0]] == i][:,
                      recommender.order[1]]
            all_items_u.append(items_u)

        # Compute MRR
        mrr = MeanReciprocalRank(results=y_pred, query=all_items_u)
        print('-> MRR (input data): %.3f' % mrr)

        # Check correctness
        self.assertGreaterEqual(mrr, 0)
        np.testing.assert_equal(y_pred, y_pred2)
        np.testing.assert_equal(y_pred, y_pred3)


    def test_CLiMF_CV(self):
        pass
        # from Orange.evaluation.testing import CrossValidation
        #
        # # Load data
        # filename = '../datasets/binary_data.tab'
        # data = Orange.data.Table(filename)
        #
        # brismf = CLiMFLearner(K=2, steps=1, alpha=0.0001, beta=0.001, verbose=False)
        # learners = [brismf]
        #
        # res = CrossValidation(data, learners, k=3)
        # mrr = MeanReciprocalRank(res)
        #
        # print("Learner  MRR")
        # for i in range(len(learners)):
        #     print(
        #         "{:8s} {:.3f}".format(learners[i].name, mrr[i]))
        #
        # self.assertIsInstance(mrr, np.ndarray)


    def test_CLiMF_exceptions(self):

        # Load data
        data = Orange.data.Table('binary_data.tab')

        # Train recommender
        learner = CLiMFLearner(K=2, steps=1, alpha=0.0, verbose=False)
        recommender = learner(data)

        self.assertWarns(
            UserWarning,
            learner,
            data
        )

        arg = 'Something that is not a table'
        self.assertRaises(TypeError, recommender, arg)


if __name__ == "__main__":
    # Test all
    #unittest.main()

    # Test single test
    suite = unittest.TestSuite()
    suite.addTest(TestCLiMF("test_CLiMF_input_data"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
