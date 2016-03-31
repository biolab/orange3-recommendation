from Orange.base import Model, Learner

__all__ = ['UserBasedLearner']

class UserBasedLearner(Learner):

    def __init__(self):
        pass

    def fit(self, X, similarity):
        """ Compute all ratings

        """
        pass
    

class UserBasedModel(Model):
    
    def recommend(self, user, top=None):
        """ Sort recomendations for user """
        pass
    
    def __str__(self):
        return 'UserBased {}'.format('---> return model')