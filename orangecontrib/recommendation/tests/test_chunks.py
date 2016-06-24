import unittest

from orangecontrib.recommendation.utils import format_data
from scipy.sparse import csr_matrix

import numpy as np

class TestChunks(unittest.TestCase):

    def test_Chunks_BuildSparseMatrix(self):
        row = np.array([0, 0, 1, 2, 2, 2])
        col = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        shape = (3, 3)
        sparse_matrix = format_data.build_sparse_matrix(row, col, data, shape)

        self.assertIsInstance(sparse_matrix, csr_matrix)


if __name__ == "__main__":
    unittest.main()