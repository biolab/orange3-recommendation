from orangecontrib.recommendation.tests.coverage import TestRatingModels
from orangecontrib.recommendation import UserAvgLearner

import unittest

__dataset__ = 'ratings.tab'


class TestUserAvg(unittest.TestCase, TestRatingModels):

    def test_input_data_continuous(self):
        learner = UserAvgLearner(verbose=True)
        super().test_input_data_continuous(learner, filename=__dataset__)

    def test_input_data_discrete(self):
        learner = UserAvgLearner()
        super().test_input_data_discrete(learner, filename='ratings_dis.tab')

    def test_pairs(self):
        learner = UserAvgLearner()
        super().test_pairs(learner, filename=__dataset__)

    def test_predict_items(self):
        learner = UserAvgLearner()
        super().test_predict_items(learner, filename=__dataset__)

    def test_swap_columns(self):
        learner = UserAvgLearner()
        super().test_swap_columns(learner, filename1='ratings_dis.tab',
                                  filename2='ratings_dis_swap.tab')

    def test_CV(self):
        learner = UserAvgLearner()
        super().test_CV(learner, filename=__dataset__)

    @unittest.skip("Skipping test")
    def test_warnings(self):
        learner = UserAvgLearner()
        super().test_warnings(learner, filename=__dataset__)


if __name__ == "__main__":
    # Test all
    unittest.main()

    # # Test single test
    # suite = unittest.TestSuite()
    # suite.addTest(TestUserAvg("test_input_data_continuous"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

