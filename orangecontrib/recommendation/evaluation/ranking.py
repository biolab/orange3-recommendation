import numpy as np

__all__ = ['ReciprocalRank', 'MeanReciprocalRank']


def ReciprocalRank(results, query):
    """Reciprocal Rank

    The RR is the multiplicative inverse of the rank of
    the first correct answer.

    Args:
        results: array
            Array with the responses. => [[4, 2, 12, 52], [3, 2, 10]]

        query: array
            Array with the query. => [[12], [10, 4]]

    Returns:
        output: List of floats

    """

    all_ranks = []
    for i in range(len(results)):

        all_rr = []
        for j in query[i]:
            rank = np.where(results[i] == j)[0]

            if len(rank) == 0:  # Check values not found
                rank = len(results[i])
            else:
                rank = rank[0]
            all_rr.append(rank)

        # Get the item best ranked (the smaller, the better; 1st, 2nd,..)
        min_rank = min(all_rr)

        rr = 1.0 / (min_rank + 1)
        all_ranks.append(rr)

    return all_ranks


def MeanReciprocalRank(results, query):
    """Mean Reciprocal Rank

    The MRR is statistic measure to evaluate processes which produce multiple
    responses to a query, sorted by probability of
    correctness.

    Args:
        results: array
            Array with the responses. => [[4, 2, 12, 52], [3, 2, 10]]

        query: array
            Array with the query. => [[12], [10, 4]]

    Returns:
        output: float

    """

    return np.mean(ReciprocalRank(results, query))

