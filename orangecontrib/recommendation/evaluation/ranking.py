import numpy as np
import sklearn.metrics as skl_metrics
from Orange.evaluation.scoring import Score

__all__ = ['ReciprocalRank', 'MeanReciprocalRank']


def ReciprocalRank(results, query):

    all_ranks = []
    for i in range(len(results)):

        all_rr = []
        for j in query[i]:
            rank = np.where(results[i] == j)[0]
            all_rr.append(rank)

        # Get the item best ranked (the smaller, the better; 1st, 2nd,..)
        min_rank = min(all_rr)
        rr = 1.0 / (min_rank + 1)
        all_ranks.append(rr)

    return all_ranks


def MeanReciprocalRank(results, query):
    return np.mean(ReciprocalRank(results, query))

