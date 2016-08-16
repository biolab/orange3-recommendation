from orangecontrib.recommendation.tests.coverage import TestRatingModels
from orangecontrib.recommendation import UserItemBaselineLearner

import unittest

__dataset__ = 'ratings.tab'


class TestUserItemBaseline(unittest.TestCase, TestRatingModels):

    def test_input_data_continuous(self):
        learner = UserItemBaselineLearner(verbose=True)
        super().test_input_data_continuous(learner, filename=__dataset__)

    def test_input_data_discrete(self):
        learner = UserItemBaselineLearner()
        super().test_input_data_discrete(learner, filename='ratings_dis.tab')

    def test_pairs(self):
        learner = UserItemBaselineLearner()
        super().test_pairs(learner, filename=__dataset__)

    def test_predict_items(self):
        learner = UserItemBaselineLearner()
        super().test_predict_items(learner, filename=__dataset__)

    def test_swap_columns(self):
        learner = UserItemBaselineLearner()
        super().test_swap_columns(learner, filename1='ratings_dis.tab',
                                  filename2='ratings_dis_swap.tab')

    def test_CV(self):
        learner = UserItemBaselineLearner()
        super().test_CV(learner, filename=__dataset__)

    @unittest.skip("Skipping test")
    def test_warnings(self):
        learner = UserItemBaselineLearner()
        super().test_warnings(learner, filename=__dataset__)

if __name__ == "__main__":
    # Test all
    unittest.main()

    # # Test single test
    # suite = unittest.TestSuite()
    # suite.addTest(TestUserItemBaseline("test_input_data_continuous"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

