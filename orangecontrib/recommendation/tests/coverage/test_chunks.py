from orangecontrib.recommendation.utils.format_data import sparse_matrix_2d
from orangecontrib.recommendation.evaluation import ReciprocalRank

from scipy.sparse import *

import unittest
import numpy as np


class TestChunks(unittest.TestCase):

    def test_Chunks_sparse_matrix(self):
        row = np.array([0, 0, 1, 2, 2, 2])
        col = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        shape = (3, 3)
        sparse_matrix = sparse_matrix_2d(row, col, data, shape, lil_matrix)

        self.assertIsInstance(sparse_matrix, lil_matrix)

    def test_Chunks_ReciprocalRank(self):
        results = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        query = np.array([[1], [5]])
        ranks = ReciprocalRank(results, query)

        self.assertEqual(len(ranks), len(query))


if __name__ == "__main__":
    # Test all
    unittest.main()

    # # Test single test
    # suite = unittest.TestSuite()
    # suite.addTest(TestChunks("test_Chunks_ReciprocalRank"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
