import os
import sys
import Orange

# Loading modules
from .base_recommendation import (ModelRecommendation as Model,
                                  LearnerRecommendation as Learner)
from .baseline.global_avg import *
from .baseline.user_avg import *
from .baseline.item_avg import *
from .baseline.user_item_baseline import *
from .rating.brismf import *
from .rating.svdplusplus import *
from .rating.trustsvd import *
from .ranking.climf import *


# Load datasets into Orange
def register_datasets():

    dataset_dir = os.path.join(os.path.dirname(__file__), 'datasets')

    if dataset_dir not in Orange.data.table.dataset_dirs:
        Orange.data.table.dataset_dirs.append(dataset_dir)


register_datasets()

