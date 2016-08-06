import numpy as np
from scipy.sparse import lil_matrix

import math


def cache_rows(matrix, key, cache):
    """
    Function to cache rows in a dictionary.
    Accessing in sparse matrices tend to be pretty slow, even for the correct
    type of structure. Therefore, this caching function is in charge of
    speeding-up the accessing.
    """

    try:
        res = cache.get(key)
        if res is None:
            if key < matrix.shape[0]:
                res = np.asarray(matrix.rows[key])
            else:
                res = []
            cache[key] = res
        return res

    except AttributeError as e:
        if isinstance(matrix, lil_matrix):
            raise AttributeError(e)
        else:
            raise TypeError(
                "'matrix' object must be a scipy.sparse.lil_matrix.")

    except Exception as e:
        raise Exception(e)


def cache_norms(matrix, indices, cache):
    """
    Vectorize function to cache norms in a numpy array.
    I tried to avoid this by using numpy.vectorize() as well as using a simple
    for loop over cache_in_array() but in both cases the code is pretty slow.
    """

    try:
        # Python philosophy:
        #      "It's easier to ask forgiveness than permission."

        # If it's not an array, try your luck
        if not isinstance(indices, np.ndarray):
            res = cache[indices]
            if res == 0:
                res = cache[indices] = math.sqrt(len(matrix.rows[indices]))
            return res

        # If the above has failed, let's hope for an array
        else:

            f_res = np.zeros(len(indices))

            # Get values
            values = cache[indices]

            # Get indices
            known_idx = values.nonzero()[0]
            unknown_idx = np.where(values == 0)[0]

            if len(known_idx) > 0:
                f_res[known_idx] = values[known_idx]

            if len(unknown_idx) > 0:
                for i, key in enumerate(indices[unknown_idx]):
                    idx = unknown_idx[i]
                    f_res[idx] = cache[key] = math.sqrt(len(matrix.rows[key]))

            return f_res

    except AttributeError as e:
        if isinstance(matrix, lil_matrix):
            raise AttributeError(e)
        else:
            raise TypeError(
                "'matrix' object must be a scipy.sparse.lil_matrix.")

    except Exception as e:
        print(e)
        raise TypeError("'indices' object must be either an 'int' or array.")