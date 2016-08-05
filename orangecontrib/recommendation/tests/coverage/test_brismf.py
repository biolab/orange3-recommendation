import Orange
from orangecontrib.recommendation.tests.coverage import TestRatingModels
from orangecontrib.recommendation import BRISMFLearner

import unittest

__dataset__ = 'ratings.tab'


class TestBRISMF(unittest.TestCase, TestRatingModels):

    def test_input_data_continuous(self):
        learner = BRISMFLearner(num_factors=2, num_iter=5, min_rating=0,
                                max_rating=5, verbose=True)
        super().test_input_data_continuous(learner, filename=__dataset__)

    def test_input_data_discrete(self):
        learner = BRISMFLearner(num_factors=2, num_iter=1)
        super().test_input_data_discrete(learner, filename='ratings_dis.tab')

    def test_pairs(self):
        learner = BRISMFLearner(num_factors=2, num_iter=1)
        super().test_pairs(learner, filename=__dataset__)

    def test_predict_items(self):
        learner = BRISMFLearner(num_factors=2, num_iter=1)
        super().test_predict_items(learner, filename=__dataset__)

    def test_CV(self):
        learner = BRISMFLearner(num_factors=2, num_iter=1)
        super().test_CV(learner, filename=__dataset__)

    def test_warnings(self):
        learner = BRISMFLearner(num_factors=2, num_iter=1, learning_rate=0.0)
        super().test_warnings(learner, filename=__dataset__)

    def test_objective(self):
        from orangecontrib.recommendation.rating.brismf import compute_loss

        # Load data
        data = Orange.data.Table(__dataset__)

        steps = [1, 10, 30]
        objectives = []
        learner = BRISMFLearner(num_factors=2, learning_rate=0.0007,
                                random_state=42, verbose=False)

        for step in steps:
            learner.num_iter = step
            recommender = learner(data)

            # Set parameters
            bias = recommender.bias
            bias_t = (bias['globalAvg'], bias['dUsers'], bias['dItems'])
            low_rank_matrices = (recommender.P, recommender.Q)
            params = (learner.lmbda, learner.bias_lmbda)

            objective = compute_loss(data, bias_t, low_rank_matrices, params)
            objectives.append(objective)

        # Assert objective values decrease
        test = list(
            map(lambda t: t[0] >= t[1], zip(objectives, objectives[1:])))
        self.assertTrue(all(test))


if __name__ == "__main__":
    # # Test all
    unittest.main()

    # # Test single test
    # suite = unittest.TestSuite()
    # suite.addTest(TestBRISMF("test_objective"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

