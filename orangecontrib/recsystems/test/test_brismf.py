import unittest

import numpy as np
from recsystems.models import brismf
import Orange

class TestBRISMF(unittest.TestCase):

    """
    def test_BRISMF_predict_items(self):
        print('\n\nPREDICT ITEMS TEST:')
        print('----------------------------')

        ratings_matrix = np.asarray([
            [5, 3, 0, 1],
            [4, 0, 0, 1],
            [1, 1, 0, 5],
            [1, 0, 0, 4],
            [0, 1, 5, 4],
        ])

        data = Orange.data.Table(ratings_matrix)
        print(ratings_matrix)

        learner = brismf.BRISMFLearner(K=5, steps=1000, verbose=True)
        recommender = learner(data)

        indices = np.array([1, 3])
        print('Indices users: ')
        print(indices)
        print('')

        print('Prediction:')
        prediction = recommender.predict_items(indices)
        print(prediction)

        self.assertAlmostEqual(0, 0)
    """

    def test_BRISMF_pairs(self):
        print('\n\nPAIRS TEST:')
        print('----------------------------')

        # Forced fit matrix to ease the visualization of the result
        ratings_matrix = np.asarray([
            [5, 3, 0, 1],
            [4, 0, 0, 1],
            [1, 1, 0, 5],
            [1, 0, 0, 4],
            [0, 1, 5, 4],
        ])

        data = Orange.data.Table(ratings_matrix)
        print(ratings_matrix)

        learner = brismf.BRISMFLearner(K=2, steps=100, verbose=True)
        recommender = learner(data)

        indices = np.array([[0, 0], [4, 3], [0, 1]])
        print('Indices pairs: ')
        print(indices)
        print('')

        print('Prediction:')
        prediction = recommender(indices)
        print(prediction)

        self.assertAlmostEqual(0, 0)

    """
        def test_BRISMF_CV(self):
            import Orange
            from Orange.evaluation.testing import CrossValidation

            filename = '/Users/salvacarrion/Documents/Programming_projects/' \
                       'PyCharm/orange3-recommendersystems/orangecontrib/' \
                       'recsystems/datasets/ratings-small.tab'

            data = Orange.data.Table(filename)

            learners = [brismf.BRISMFLearner(K=2, steps=100)]
            res = CrossValidation(data, learners)
            asd = 3
    """

if __name__ == "__main__":
    unittest.main()

