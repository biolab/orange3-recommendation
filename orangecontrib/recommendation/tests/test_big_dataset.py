import Orange
from Orange.evaluation.testing import CrossValidation
from orangecontrib.recommendation import *

from sklearn.metrics import mean_squared_error

import math
import time


def test_learners():
    start = time.time()

    # Load data
    data = Orange.data.Table('movielens100k.tab')
    print('- Loading time: %.3fs' % (time.time() - start))


    start = time.time()
    learner = GlobalAvgLearner()
    recommender = learner(data)
    print('- Time (GlobalAvgLearner): %.3fs' % (time.time() - start))
    rmse = math.sqrt(mean_squared_error(data.Y, recommender(data)))
    print('- RMSE (GlobalAvgLearner): %.3f' % rmse)

    start = time.time()
    learner = ItemAvgLearner()
    recommender = learner(data)
    print('- Time (ItemAvgLearner): %.3fs' % (time.time() - start))
    rmse = math.sqrt(mean_squared_error(data.Y, recommender(data)))
    print('- RMSE (ItemAvgLearner): %.3f' % rmse)

    start = time.time()
    learner = UserAvgLearner()
    recommender = learner(data)
    print('- Time (UserAvgLearner): %.3fs' % (time.time() - start))
    rmse = math.sqrt(mean_squared_error(data.Y, recommender(data)))
    print('- RMSE (UserAvgLearner): %.3f' % rmse)

    start = time.time()
    learner = UserItemBaselineLearner()
    recommender = learner(data)
    print('- Time (UserItemBaselineLearner): %.3fs' % (time.time() - start))
    rmse = math.sqrt(mean_squared_error(data.Y, recommender(data)))
    print('- RMSE (UserItemBaselineLearner): %.3f' % rmse)

    start = time.time()
    learner = BRISMFLearner(K=15, steps=10, alpha=0.07, beta=0.1, verbose=False)
    recommender = learner(data)
    print('- Time (BRISMFLearner): %.3fs' % (time.time() - start))
    rmse = math.sqrt(mean_squared_error(data.Y, recommender(data)))
    print('- RMSE (BRISMFLearner): %.3f' % rmse)

    start = time.time()
    learner = SVDPlusPlusLearner(K=15, steps=10, alpha=0.07, beta=0.1, verbose=False)
    recommender = learner(data)
    print('- Time (SVDPlusPlusLearner): %.3fs' % (time.time() - start))
    rmse = math.sqrt(mean_squared_error(data.Y, recommender(data)))
    print('- RMSE (SVDPlusPlusLearner): %.3f' % rmse)


def test_CV():

    # Load data
    data = Orange.data.Table('filmtrust/ratings_small.tab')
    trust = Orange.data.Table('filmtrust/trust_small.tab')

    # Learners
    global_avg = GlobalAvgLearner()
    items_avg = ItemAvgLearner()
    users_avg = UserAvgLearner()
    useritem_baseline = UserItemBaselineLearner()
    brismf = BRISMFLearner(K=15, steps=10, alpha=0.07, beta=0.1)
    svdpp = SVDPlusPlusLearner(K=15, steps=10, alpha=0.007, beta=0.1)
    trustsvd = TrustSVDLearner(K=15, steps=10, alpha=0.007, beta=0.1,
                               beta_trust=0.05, trust=trust)
    learners = [global_avg, items_avg, users_avg, useritem_baseline,
                brismf, svdpp, trustsvd]

    res = CrossValidation(data, learners, k=5)
    rmse = Orange.evaluation.RMSE(res)
    r2 = Orange.evaluation.R2(res)

    print("Learner  RMSE  R2")
    for i in range(len(learners)):
        print(
            "{:8s} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], r2[i]))

if __name__ == "__main__":
    pass
    test_learners()