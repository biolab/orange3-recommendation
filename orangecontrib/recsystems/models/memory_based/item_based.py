from Orange.base import Model, Learner

__all__ = ['ItemBasedLearner']

class ItemBasedLearner(Learner):

    def __init__(self):
        pass

    def fit(self, X, similarity):
        """ Compute all ratings

        """
        pass


class ItemBasedModel(Model):

    def recommend(self, user, top=None):
        """ Sort recomendations for user """
        pass

    def __str__(self):
        return 'ItemBased {}'.format('---> return model')