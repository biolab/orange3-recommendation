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
                   'recsystems/datasets/users-movies-toy.tab'

        data = Orange.data.Table(filename)
        #subdata = data[:3]
        learners = [brismf.BRISMFLearner(K=2, steps=100)]
        res = CrossValidation(data, learners)
        asd = 3

    """
    def test_BRISMF_input_data(self):
        import Orange
        from Orange.evaluation.testing import CrossValidation

        filename = '/Users/salvacarrion/Documents/Programming_projects/' \
                   'PyCharm/orange3-recommendersystems/orangecontrib/' \
                   'recsystems/datasets/users-movies-toy.tab'

        data = Orange.data.Table(filename)

        learner = brismf.BRISMFLearner(K=2, steps=100)

        #newData = learner.build_sparse_matrix(data)
        recommender = learner(data)

        prediction = recommender(data[:3])
        print(prediction)
    """


if __name__ == "__main__":
    unittest.main()

