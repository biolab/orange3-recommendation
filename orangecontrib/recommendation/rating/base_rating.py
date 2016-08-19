from orangecontrib.recommendation import Learner, Model

import numpy as np

__all__ = ["LearnerRecommendation", "ModelRecommendation"]


class ModelRecommendation(Model):

    def predict_on_range(self, predictions):
        # Just for modeling ratings with latent factors
        try:
            if self.min_rating is not None:
                predictions[predictions < self.min_rating] = self.min_rating

            if self.max_rating is not None:
                predictions[predictions > self.max_rating] = self.max_rating
        finally:
            return predictions

    def fix_predictions(self, X, predictions, bias):
        idxs_users_missing, idxs_items_missing = self.indices_missing

        # Set average when neither the user nor the item exist
        g_avg = bias['globalAvg']
        common_indices = np.intersect1d(idxs_users_missing, idxs_items_missing)
        predictions[common_indices] = g_avg

        # Only users exist (return average + {dUser})
        if 'dUsers' in bias:
            missing_users = np.setdiff1d(idxs_users_missing, common_indices)
            if len(missing_users) > 0:
                user_idxs = X[missing_users, self.order[0]]
                predictions[missing_users] = g_avg + bias['dUsers'][user_idxs]

        # Only items exist (return average + {dItem})
        if 'dItems' in bias:
            missing_items = np.setdiff1d(idxs_items_missing, common_indices)
            if len(missing_items) > 0:
                item_idxs = X[missing_items, self.order[1]]
                predictions[missing_items] = g_avg + bias['dItems'][item_idxs]

        return predictions


class LearnerRecommendation(Learner):

    def __init__(self, preprocessors=None, verbose=False, min_rating=None,
                 max_rating=None):
        self.min_rating = min_rating
        self.max_rating = max_rating
        super().__init__(preprocessors=preprocessors, verbose=verbose)

    def prepare_model(self, model):
        model.min_rating = self.min_rating
        model.max_rating = self.max_rating
        return super().prepare_model(model)