from orangecontrib.recommendation import Learner, Model
from Orange.data import Table

import numpy as np

__all__ = ["LearnerRecommendation", "ModelRecommendation"]


class ModelRecommendation(Model):

    def __call__(self, *args, **kwargs):
        """
        We need to override the __call__ of the base.model because it transforms
        the output to 'argmax(probabilities=X)'
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

    def compute_objective(self):
        pass


class LearnerRecommendation(Learner):

    def fit_model(self):
        pass

    def fit_base(self, data):
        """This function calls the fit method.

        Args:
            data: Orange.data.Table

        Returns:
            Model

        """
        return self.fit_model(data)
