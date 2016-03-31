import unittest

import numpy as np

#from Orange.recommenders.memory_based import (UserBased, ItemBased)
from Orange.recommenders.model_based import BRISMF


class RecommendersTest(unittest.TestCase):

    """
    def test_UserBased(self):
        ratings_matrix = {}

        recommender = UserBased()
        recommender.fit(data=ratings_matrix, similarity=None)
        prediction = recommender.recommend(user_id=5, top=5)

        correct = [0.11893, 0.10427, 0.13117, 0.14650, 0.05973]
        self.assertAlmostEqual(prediction, correct, decimal=5)


    def test_ItemBased(self):
        ratings_matrix = {}

        recommender = ItemBased()
        recommender.fit(data=ratings_matrix, similarity=None)
        prediction = recommender.recommend(user_id=5, top=5)

def test_BRISMF(self):
    data = {#MATRIX#}
    learner = Orange.recommenders.model_based.BRISMFLearner()
    recommender = learner(data)

    prediction = recommender.predict(user=1, sort=False, top=None)

    correct = [0.11893, 0.10427, 0.13117, 0.14650, 0.05973]
    self.assertAlmostEqual(prediction, correct, decimal=5)
    """

    def test_BRISMF(self):
        """
        ratings_matrix = np.array([
            [2, 0, 0, 4, 5, 0],
            [5, 0, 4, 0, 0, 1],
            [0, 0, 5, 0, 2, 0],
            [0, 1, 0, 5, 0, 4],
            [0, 0, 4, 0, 0, 2],
            [4, 5, 0, 1, 0, 0]
        ])
        """


        ratings_matrix = np.asarray([
            [5, 3, 0, 1],
            [4, 0, 0, 1],
            [1, 1, 0, 5],
            [1, 0, 0, 4],
            [0, 1, 5, 4],
        ])

        recommender = BRISMF()
        #print(ratings_matrix)
        # print('')
        recommender.fit(ratings_matrix)
        prediction = recommender.recommend(user=1, sort=False, top=None)
        #print('')
        #print(prediction[:, 1].T)
        correct = np.array([4,  1,  3,  1])
        np.testing.assert_almost_equal(
            np.round(np.round(prediction[:, 1])), np.round(correct))

"""
if __name__ == "__main__":
    test_recommender = RecommendersTest(unittest.TestCase)
    test_recommender.test_BRISMF()
"""