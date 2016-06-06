import numpy as np
from scipy import sparse

import Orange

data = Orange.data.Table("housing.tab")

lin = Orange.regression.linear.LinearRegressionLearner()
rf = Orange.regression.random_forest.RandomForestRegressionLearner()
rf.name = "rf"
ridge = Orange.regression.RidgeRegressionLearner()
mean = Orange.regression.MeanLearner()

learners = [lin, rf, ridge, mean]

res = Orange.evaluation.CrossValidation(data, learners, k=5)
rmse = Orange.evaluation.RMSE(res)
r2 = Orange.evaluation.R2(res)

print("Learner  RMSE  R2")
for i in range(len(learners)):
    print("{:8s} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], r2[i]))

from recommendation import brismf
from Orange.evaluation.testing import CrossValidation

filename_dense = '/Users/salvacarrion/Documents/Programming_projects/' \
           'PyCharm/orange3-recommendersystems/orangecontrib/' \
           'recommendation/datasets/ratings-small.tab'

filename_sparse = '/Users/salvacarrion/Documents/Programming_projects/' \
                  'PyCharm/orange3-recommendersystems/orangecontrib/' \
                  'recommendation/datasets/ratings.csv'

data = Orange.data.Table(filename_dense)

# Convert NaNs to zero
where_are_NaNs = np.isnan(data.X)
data.X[where_are_NaNs] = 0
data.X = sparse.csr_matrix(data.X)

learners = [brismf.BRISMFLearner(K=2, steps=100)]

res = Orange.evaluation.CrossValidation(data, learners, k=5)
rmse = Orange.evaluation.RMSE(res)
r2 = Orange.evaluation.R2(res)

print("Learner  RMSE  R2")
for i in range(len(learners)):
    print("{:8s} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], r2[i]))


asd = 23

