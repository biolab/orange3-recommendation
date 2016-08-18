import Orange
from orangecontrib.recommendation.tests.coverage import TestRatingModels
from orangecontrib.recommendation import TrustSVDLearner
from orangecontrib.recommendation.optimizers import *

import unittest

__dataset__ = 'filmtrust/trust_small.tab'
__trust_dataset__ = 'filmtrust/trust_small.tab'
__optimizers__ = [SGD(), Momentum(momentum=0.9),
                  NesterovMomentum(momentum=0.9), AdaGrad(),
                  RMSProp(rho=0.9), AdaDelta(rho=0.95),
                  Adam(beta1=0.9, beta2=0.999)]


class TestTrustSVD(unittest.TestCase, TestRatingModels):

    def test_input_data_continuous(self, *args):
        trust = Orange.data.Table(__trust_dataset__)
        learner = TrustSVDLearner(num_factors=2, num_iter=1, trust=trust,
                                  verbose=2)

        # Test SGD optimizers too
        for opt in __optimizers__:
            learner.optimizer = opt
            print(learner.optimizer)
            super().test_input_data_continuous(learner, filename=__dataset__)

    @unittest.skip("Skipping test")
    def test_input_data_discrete(self, *args):
        trust = Orange.data.Table(__trust_dataset__)
        learner = TrustSVDLearner(num_factors=2, num_iter=1, trust=trust)
        super().test_input_data_discrete(learner, filename='ratings_dis.tab')

    def test_pairs(self, *args):
        trust = Orange.data.Table(__trust_dataset__)
        learner = TrustSVDLearner(num_factors=2, num_iter=1, trust=trust)
        super().test_pairs(learner, filename=__dataset__)

    def test_predict_items(self, *args):
        trust = Orange.data.Table(__trust_dataset__)
        learner = TrustSVDLearner(num_factors=2, num_iter=1, trust=trust)
        super().test_predict_items(learner, filename=__dataset__)

    def test_swap_columns(self, *args):
        trust = Orange.data.Table(__trust_dataset__)
        learner = TrustSVDLearner(num_factors=2, num_iter=1, trust=trust,
                                  random_state=42)
        super().test_swap_columns(learner, filename1='ratings_dis.tab',
                                  filename2='ratings_dis_swap.tab')

    def test_CV(self, *args):
        trust = Orange.data.Table(__trust_dataset__)
        learner = TrustSVDLearner(num_factors=2, num_iter=1, trust=trust)
        super().test_CV(learner, filename=__dataset__)

    def test_warnings(self, *args):
        trust = Orange.data.Table(__trust_dataset__)
        learner = TrustSVDLearner(num_factors=2, num_iter=1,
                                  learning_rate=0.0, trust=trust)
        super().test_warnings(learner, filename=__dataset__)

    def test_objective(self):
        from orangecontrib.recommendation.rating.trustsvd import compute_loss

        # Load data
        data = Orange.data.Table(__dataset__)
        trust = Orange.data.Table(__trust_dataset__)

        steps = [1, 10, 30]
        objectives = []
        learner = TrustSVDLearner(num_factors=2, learning_rate=0.0007,
                                  random_state=42, trust=trust, verbose=False)

        for step in steps:
            learner.num_iter = step
            recommender = learner(data)

            # Set parameters
            data_t = (data, learner.trust)
            bias = recommender.bias
            bias_t = (bias['globalAvg'], bias['dUsers'], bias['dItems'])
            low_rank_matrices = (recommender.P, recommender.Q, recommender.Y,
                                 recommender.W)
            params = (learner.lmbda, learner.bias_lmbda, learner.social_lmbda)

            objective = compute_loss(data_t, bias_t, low_rank_matrices, params)
            objectives.append(objective)

        # Assert objective values decrease
        test = list(
            map(lambda t: t[0] >= t[1], zip(objectives, objectives[1:])))
        self.assertTrue(all(test))

    def test_outputs(self):
        # Load data
        data = Orange.data.Table(__dataset__)
        trust = Orange.data.Table(__trust_dataset__)
        learner = TrustSVDLearner(num_factors=2, num_iter=1, trust=trust)

        # Train recommender
        recommender = learner(data)

        # Check tables P, Q, Y and W
        P = recommender.getPTable()
        Q = recommender.getQTable()
        Y = recommender.getYTable()
        W = recommender.getWTable()

        diff = len({P.X.shape[1], Q.X.shape[1], Y.X.shape[1], W.X.shape[1]})
        self.assertEqual(diff, 1)

if __name__ == "__main__":
    # Test all
    unittest.main()

    # # Test single test
    # suite = unittest.TestSuite()
    # suite.addTest(TestTrustSVD("test_input_data_continuous"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

