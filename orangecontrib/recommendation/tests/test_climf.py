# import os
# import math
# import unittest
#
# import Orange
# from orangecontrib.recommendation import CLiMFLearner
#
# import numpy as np
# from sklearn.metrics import mean_squared_error
#
# class TestCLiMF(unittest.TestCase):
#
#     def test_CLiMF_predict_items(self):
#         # Load data
#         filename = '../datasets/binary_data.tab'
#         data = Orange.data.Table(filename)
#
#         # Train recommender
#         learner = CLiMFLearner(K=10, steps=50, beta=0.02, verbose=False)
#         recommender = learner(data)
#
#         # Compute predictions
#         prediction = recommender.predict_items()
#         y_pred = prediction[data.X[:, recommender.order[0]],
#                             data.X[:, recommender.order[1]]]
#
#         # Compute RMSE
#         rmse = math.sqrt(mean_squared_error(data.Y, y_pred))
#         print('-> RMSE (predict items): %.3f' % rmse)
#
#         # Check correctness
#         self.assertGreaterEqual(rmse, 0)
#
#
#     def test_CLiMF_input_data(self):
#         # Load data
#         filename = '../datasets/binary_data.tab'
#         data = Orange.data.Table(filename)
#
#         # Train recommender
#         learner = CLiMFLearner(K=5, steps=10, alpha=0.07, beta=0.02, verbose=False)
#         recommender = learner(data)
#
#         # Compute predictions
#         y_pred = recommender(data)
#
#         # Compute RMSE
#         rmse = math.sqrt(mean_squared_error(data.Y, y_pred))
#         print('-> RMSE (input data): %.3f' % rmse)
#
#         # Check correctness
#         self.assertGreaterEqual(rmse, 0)
#
#
#     def test_CLiMF_pairs(self):
#         # Load data
#         filename = '../datasets/binary_data.tab'
#         data = Orange.data.Table(filename)
#
#         # Train recommender
#         learner = CLiMFLearner(K=5, steps=10, alpha=0.07, beta=0.02, verbose=False)
#         recommender = learner(data)
#
#         # Create indices to test
#         sample_size = 10
#         num_users, num_items = recommender.shape
#         idx_users = np.random.randint(0, num_users, size=sample_size)
#         idx_items = np.random.randint(0, num_items, size=sample_size)
#         indices = np.column_stack((idx_users, idx_items))
#
#         # Compute predictions
#         y_pred = recommender(indices)
#         print('-> Same number? (pairs): %r' % (len(y_pred) == sample_size))
#
#         # Check correctness
#         self.assertEqual(len(y_pred), sample_size)
#
#
#     def test_CLiMF_CV(self):
#         from Orange.evaluation.testing import CrossValidation
#
#         # Load data
#         filename = '../datasets/binary_data.tab'
#         data = Orange.data.Table(filename)
#
#         brismf = CLiMFLearner(K=5, steps=50, beta=0.02, verbose=False)
#         learners = [brismf]
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
#         self.assertIsInstance(rmse, np.ndarray)
#
# if __name__ == "__main__":
#     # Test all
#     unittest.main()
#
#     # Test single test
#     # suite = unittest.TestSuite()
#     # suite.addTest(TestCLiMF("test_CLiMF_input_data"))
#     # runner = unittest.TextTestRunner()
#     # runner.run(suite)
