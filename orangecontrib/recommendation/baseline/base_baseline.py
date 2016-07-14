from orangecontrib.recommendation import Learner, Model

__all__ = ["LearnerRecommendation", "ModelRecommendation"]


class ModelRecommendation(Model):
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
