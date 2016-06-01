import unittest

import numpy as np
from recsystems.models import brismf


class TestBRISMF(unittest.TestCase):

    """
    def test_BRISMF1(self):
        import Orange

        filename = '/Users/salvacarrion/Documents/Programming_projects/' \
                    'PyCharm/orange3-recommendersystems/orangecontrib/' \
                    'recsystems/datasets/ratings-small.tab'

        data = Orange.data.Table(filename)

        learner = brismf.BRISMFLearner(K=2, steps=100, verbose=True)
        recommender = learner(data)


        prediction = recommender.predict_items(user=1, sort=False, top=None)
        print(prediction[:, 1].T)
    """


    def test_BRISMF2(self):
        import Orange

        filename = '/Users/salvacarrion/Documents/Programming_projects/' \
                    'PyCharm/orange3-recommendersystems/orangecontrib/' \
                    'recsystems/datasets/ratings-small.tab'

        data = Orange.data.Table(filename)

        learner = brismf.BRISMFLearner(K=2, steps=100, verbose=True)
        recommender = learner(data)

        indices = np.array([1, 5])

        prediction = recommender(indices)
        print(prediction[:, 1].T)

    """

       def test_BRISMF4(self):
        import Orange
        from Orange.evaluation.testing import CrossValidation

        filename = '/Users/salvacarrion/Documents/Programming_projects/' \
                   'PyCharm/orange3-recommendersystems/orangecontrib/' \
                   'recsystems/datasets/ratings-small.tab'

        data = Orange.data.Table(filename)

        learners = [brismf.BRISMFLearner(K=2, steps=100)]
        res = CrossValidation(data, learners)
        asd = 3


    def test_BRISMF4_WRONG(self):
        ratings_matrix = np.array([
            [2, 0, 0, 4, 5, 0],
            [5, 0, 4, 0, 0, 1],
            [0, 0, 5, 0, 2, 0],
            [0, 1, 0, 5, 0, 4],
            [0, 0, 4, 0, 0, 2],
            [4, 5, 0, 1, 0, 0]
        ])

        ratings_matrix = np.asarray([
            [5, 3, 0, 1],
            [4, 0, 0, 1],
            [1, 1, 0, 5],
            [1, 0, 0, 4],
            [0, 1, 5, 4],
        ])

        recommender = brismf()
        # print(ratings_matrix)
        # print('')
        recommender.fit(ratings_matrix)
        prediction = recommender.recommend(user=1, sort=False, top=None)
        # print('')
        # print(prediction[:, 1].T)
        correct = np.array([4, 1, 3, 1])
        np.testing.assert_almost_equal(
            np.round(np.round(prediction[:, 1])), np.round(correct))
    """

if __name__ == "__main__":
    unittest.main()

