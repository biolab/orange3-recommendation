from orangecontrib.recommendation import Learner, Model
from orangecontrib.recommendation.utils.format_data import *
from orangecontrib.recommendation.evaluation.ranking import MeanReciprocalRank
from scipy.sparse import lil_matrix

import numpy as np

__all__ = ["LearnerRecommendation", "ModelRecommendation"]


class ModelRecommendation(Model):

    def __call__(self, *args, **kwargs):
        """
        We need to override the __call__ of the base.model because Orange3
        transforms the output of model.predict() to 'argmax(probabilities=X)'
        """

        data = args[0]
        top_k = None
        if 'top_k' in kwargs:  # Check if this parameters exists
            top_k = kwargs['top_k']

        if isinstance(data, np.ndarray):
            prediction = self.predict(X=data, top_k=top_k)
        elif isinstance(data, Table):
            prediction = self.predict(X=data.X.astype(int), top_k=top_k)
        else:
            raise TypeError("Unrecognized argument (instance of '{}')"
                            .format(type(data).__name__))

        return prediction

    def compute_mrr(self, data, users, queries=None):

        # Check data type
        if isinstance(data, lil_matrix):
            pass
        elif isinstance(data, Table):
            # Preprocess Orange.data.Table and transform it to sparse
            data, order, shape = preprocess(data)
            data = table2sparse(data, shape, order, m_type=lil_matrix)
        else:
            raise TypeError('Invalid data type')

        # Make predictions
        y_pred = self(users)

        # Get relevant items for the user[i]
        if queries is None:
            queries = []
            add_items = queries.append
            for u in users:
                add_items(np.asarray(data.rows[u]))

        # Compute Mean Reciprocal Rank (MRR)
        mrr = MeanReciprocalRank(results=y_pred, query=queries)
        return mrr, queries


class LearnerRecommendation(Learner):
    pass
