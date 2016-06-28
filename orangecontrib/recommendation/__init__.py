from .base_recommendation import (ModelRecommendation as Model,
                                  LearnerRecommendation as Learner)
from .global_avg import *
from .user_avg import *
from .item_avg import *
from .user_item_baseline import *
from .brismf import *
from .climf import *


# Load datasets into Orange
import  Orange
from os.path import abspath, join, dirname
filename = abspath(join(dirname(__file__), 'datasets'))
Orange.data.table.dataset_dirs.insert(0, filename)