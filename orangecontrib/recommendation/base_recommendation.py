from Orange.base import Learner, Model

__all__ = ["LearnerRecommendation", "ModelRecommendation"]

class ModelRecommendation(Model):
    pass

class LearnerRecommendation(Learner):
    __returns__ = ModelRecommendation