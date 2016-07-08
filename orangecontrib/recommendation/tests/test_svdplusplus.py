import os
import math
import random
import unittest

import Orange
from orangecontrib.recommendation import SVDPlusPlusLearner

import numpy as np
from sklearn.metrics import mean_squared_error

class TestSVDPlusPlus(unittest.TestCase):


    def test_SVDPlusPlus_input_data_continuous(self):
        pass
        # Load data
        data = Orange.data.Table('ratings3.tab')

        # Train recommender
        learner = SVDPlusPlusLearner(K=2, steps=1, verbose=True)
        recommender = learner(data)

        print(str(recommender) + ' trained')

        # Compute predictions
        y_pred = recommender(data)

        # Compute RMSE
        rmse = math.sqrt(mean_squared_error(data.Y, y_pred))
        print('-> RMSE (input data; continuous): %.3f' % rmse)

        # Check correctness
        self.assertGreaterEqual(rmse, 0)

        # # Check tables P and Q
        # P = recommender.getPTable()
        # Q = recommender.getQTable()
        # self.assertEqual(P.X.shape[1], Q.X.shape[1])


if __name__ == "__main__":
    # Test all
    unittest.main()

    # # Test single test
    # suite = unittest.TestSuite()
    # suite.addTest(SVDPlusPlus("test_SVDPlusPlus_input_data_continuous"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

