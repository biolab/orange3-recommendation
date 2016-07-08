# import os
# import math
# import time
# import unittest
#
# import Orange
# from orangecontrib.recommendation import *
#
# import numpy as np
# from sklearn.metrics import mean_squared_error
#
# class TestBigDataset(unittest.TestCase):
#     pass
#
#     def test_learners(self):
#         start = time.time()
#
#         # Load data
#         data = Orange.data.Table('movielens100k.tab')
#         print('- Loading time: %.3fs' % (time.time() - start))
#
#
#         start = time.time()
#         learner = GlobalAvgLearner()
#         recommender = learner(data)
#         print('- Time (GlobalAvgLearner): %.3fs' % (time.time() - start))
#         rmse = math.sqrt(mean_squared_error(data.Y, recommender(data)))
#         print('- RMSE (GlobalAvgLearner): %.3f' % rmse)
#
#         start = time.time()
#         learner = ItemAvgLearner()
#         recommender = learner(data)
#         print('- Time (ItemAvgLearner): %.3fs' % (time.time() - start))
#         rmse = math.sqrt(mean_squared_error(data.Y, recommender(data)))
#         print('- RMSE (ItemAvgLearner): %.3f' % rmse)
#
#         start = time.time()
#         learner = UserAvgLearner()
#         recommender = learner(data)
#         print('- Time (UserAvgLearner): %.3fs' % (time.time() - start))
#         rmse = math.sqrt(mean_squared_error(data.Y, recommender(data)))
#         print('- RMSE (UserAvgLearner): %.3f' % rmse)
#
#         start = time.time()
#         learner = UserItemBaselineLearner()
#         recommender = learner(data)
#         print('- Time (UserItemBaselineLearner): %.3fs' % (time.time() - start))
#         rmse = math.sqrt(mean_squared_error(data.Y, recommender(data)))
#         print('- RMSE (UserItemBaselineLearner): %.3f' % rmse)
#
#         start = time.time()
#         learner = BRISMFLearner(K=15, steps=10, alpha=0.07, beta=0.1, verbose=False)
#         recommender = learner(data)
#         print('- Time (BRISMFLearner): %.3fs' % (time.time() - start))
#         rmse = math.sqrt(mean_squared_error(data.Y, recommender(data)))
#         print('- RMSE (BRISMFLearner): %.3f' % rmse)
#
#         start = time.time()
#         learner = SVDPlusPlusLearner(K=10, steps=10, alpha=0.07, beta=0.1,
#                                      verbose=False)
#         recommender = learner(data)
#         print('- Time (SVDPlusPlusLearner): %.3fs' % (time.time() - start))
#         rmse = math.sqrt(mean_squared_error(data.Y, recommender(data)))
#         print('- RMSE (SVDPlusPlusLearner): %.3f' % rmse)
#
#         self.assertEqual(1, 1)
#
#
#     def test_CV(self):
#         from Orange.evaluation.testing import CrossValidation
#         # Load data
#         filename = '/Users/salvacarrion/Desktop/big_datasets/MovieLens100K.tab'
#         data = Orange.data.Table(filename)
#
#         global_avg = GlobalAvgLearner()
#         items_avg = ItemAvgLearner()
#         users_avg = UserAvgLearner()
#         useritem_baseline = UserItemBaselineLearner()
#         brismf = BRISMFLearner(K=15, steps=10, alpha=0.07, beta=0.1)
#         learners = [global_avg, items_avg, users_avg, useritem_baseline, brismf]
#
#         res = CrossValidation(data, learners, k=5)
#         rmse = Orange.evaluation.RMSE(res)
#         r2 = Orange.evaluation.R2(res)
#
#         print("Learner  RMSE  R2")
#         for i in range(len(learners)):
#             print(
#                 "{:8s} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], r2[i]))
#
#         self.assertEqual(1, 1)
#
#
# if __name__ == "__main__":
#     # Test all
#     # unittest.main()
#
#     # Test single test
#     suite = unittest.TestSuite()
#     suite.addTest(TestBigDataset("test_learners"))
#     runner = unittest.TextTestRunner()
#     runner.run(suite)