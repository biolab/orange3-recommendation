import Orange
from orangecontrib.recommendation import *


import time
import math

from sklearn.metrics import mean_squared_error

def test_speed():
    start = time.time()

    # Load data
    filename = '../datasets/MovieLensOrange.tab'
    data = Orange.data.Table(filename)

    print('- Loading time: %.3fs' % (time.time() - start))


    #
    # start = time.time()
    # learner = GlobalAvgLearner()
    # recommender = learner(data)
    # print('- Time (GlobalAvgLearner): %.3fs' % (time.time() - start))
    #
    # start = time.time()
    # learner = ItemAvgLearner()
    # recommender = learner(data)
    # print('- Time (ItemAvgLearner): %.3fs' % (time.time() - start))
    #
    # start = time.time()
    # learner = UserAvgLearner()
    # recommender = learner(data)
    # print('- Time (UserAvgLearner): %.3fs' % (time.time() - start))

    # start = time.time()
    # learner = UserItemBaselineLearner()
    # recommender = learner(data)
    # print('- Time (UserItemBaselineLearner): %.3fs' % (time.time() - start))

    start = time.time()
    learner = BRISMFLearner(K=5, steps=10, alpha=0.07, beta=0.01, verbose=True)
    recommender = learner(data)
    print('- Time (BRISMFLearner): %.3fs' % (time.time() - start))


def test_training_set():
    # Load data
    filename = '../datasets/MovieLensOrange.tab'
    data = Orange.data.Table(filename)



    learner = GlobalAvgLearner()
    recommender = learner(data)
    rmse = math.sqrt(mean_squared_error(data.Y, recommender(data)))
    print('- RMSE (GlobalAvgLearner): %.3f' % rmse)

    start = time.time()
    learner = ItemAvgLearner()
    recommender = learner(data)
    rmse = math.sqrt(mean_squared_error(data.Y, recommender(data)))
    print('- RMSE (ItemAvgLearner): %.3f' % rmse)

    start = time.time()
    learner = UserAvgLearner()
    recommender = learner(data)
    rmse = math.sqrt(mean_squared_error(data.Y, recommender(data)))
    print('- RMSE (UserAvgLearner): %.3f' % rmse)

    start = time.time()
    learner = UserItemBaselineLearner()
    recommender = learner(data)
    rmse = math.sqrt(mean_squared_error(data.Y, recommender(data)))
    print('- RMSE (UserItemBaselineLearner): %.3f' % rmse)
    exit()
    start = time.time()
    learner = BRISMFLearner(K=5, steps=10, alpha=0.07, beta=0.1, verbose=False)
    recommender = learner(data)
    rmse = math.sqrt(mean_squared_error(data.Y, recommender(data)))
    print('- RMSE (BRISMFLearner): %.3f' % rmse)

def test_CV():
    from Orange.evaluation.testing import CrossValidation
    # Load data
    filename = '../datasets/MovieLensOrange.tab'
    data = Orange.data.Table(filename)

    global_avg = GlobalAvgLearner()
    items_avg = ItemAvgLearner()
    users_avg = UserAvgLearner()
    user_item_baseline = UserItemBaselineLearner()
    brismf = BRISMFLearner(K=15, steps=10, alpha=0.07, beta=0.02, verbose=True)
    learners = [global_avg, items_avg, users_avg, user_item_baseline, brismf]

    res = CrossValidation(data, learners, k=5)
    rmse = Orange.evaluation.RMSE(res)
    r2 = Orange.evaluation.R2(res)

    print("Learner  RMSE  R2")
    for i in range(len(learners)):
        print(
            "{:8s} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], r2[i]))



if __name__ == "__main__":
    #test_speed()
    #test_training_set()
    test_CV()