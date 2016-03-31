"""
CLiMF Collaborative Less-is-More Filtering, a variant of latent factor CF
which optimises a lower bound of the smoothed reciprocal rank of "relevant"
items in ranked recommendation lists.  The intention is to promote diversity
as well as accuracy in the recommendations.  The method assumes binary
relevance data, as for example in friendship or follow relationships.

CLiMF: Learning to Maximize Reciprocal Rank with Collaborative Less-is-More Filtering
Yue Shi, Martha Larson, Alexandros Karatzoglou, Nuria Oliver, Linas Baltrunas, Alan Hanjalic
ACM RecSys 2012
"""

from math import exp, log, pow, e
import numpy as np
import random
from scipy import sparse, io

class CLiMF:

    def __init__(self, K=10, lmda=0.001, gamma=0.0001, max_iters=10):
        self.K = K
        self.lmda = lmda
        self.gamma = gamma
        self.max_iters = max_iters
        self.U = None
        self.V = None

    def tensor(self, x, beta):
        return pow(e, -beta*x)


    def g(self, x):
        """sigmoid function"""
        return 1/(1+exp(-x))


    def dg(self, x):
        """derivative of sigmoid function"""
        return exp(x)/(1+exp(x))**2


    def precompute_f(self, data,U,V,i):
        """precompute f[j] = <U[i],V[j]>
        params:
          data: scipy csr sparse matrix containing user->(item,count)
          U   : user factors
          V   : item factors
          i   : item of interest
        returns:
          dot products <U[i],V[j]> for all j in data[i]
        """
        items = data[i].indices
        f = dict((j,np.dot(U[i],V[j])) for j in items)
        return f


    def objective(self, data,U,V,lbda):
        """compute objective function F(U,V)
        params:
          data: scipy csr sparse matrix containing user->(item,count)
          U   : user factors
          V   : item factors
          lbda: regularization constant lambda
        returns:
          current value of F(U,V)
        """
        F = -0.5*lbda*(np.sum(U*U)+np.sum(V*V))
        for i in range(len(U)):
            f = self.precompute_f(data,U,V,i)
            for j in f:
                F += log(self.g(f[j]))
                for k in f:
                    inv_g1 = 1-self.g(f[k]-f[j])
                    F += log(inv_g1)
        return F


    def update(self, data,U,V,lbda,gamma):
        """update user/item factors using stochastic gradient ascent
        params:
          data : scipy csr sparse matrix containing user->(item,count)
          U    : user factors
          V    : item factors
          lbda : regularization constant lambda
          gamma: learning rate
        """
        for i in range(len(U)):
            dU = -lbda*U[i]
            f = self.precompute_f(data,U,V,i)
            for j in f:
                dV = self.g(-f[j])-lbda*V[j]
                for k in f:
                    inv_g1 = 1-self.g(f[k]-f[j])
                    inv_g2 = 1-self.g(f[j]-f[k])
                    dV += self.dg(f[j]-f[k])*(1/(inv_g1)-1/(inv_g2))*U[i]

                V[j] += gamma*dV
                dU += self.g(-f[j])*V[j]
                for k in f:
                    dU += (V[j]-V[k])*self.dg(f[k]-f[j])/(1-self.g(f[k]-f[j]))
            U[i] += gamma*dU


    def compute_mrr(self, data,U,V,test_users=None):
        """compute average Mean Reciprocal Rank of data according to factors
        params:
          data      : scipy csr sparse matrix containing user->(item,count)
          U         : user factors
          V         : item factors
          test_users: optional subset of users over which to compute MRR
        returns:
          the mean MRR over all users in data
        """
        mrr = []
        if test_users is None:
            test_users = range(len(U))
        for ix,i in enumerate(test_users):
            items = set(data[i].indices)
            predictions = np.sum(np.tile(U[i],(len(V),1))*V,axis=1)
            for rank,item in enumerate(np.argsort(predictions)[::-1]):
                if item in items:
                    mrr.append(1.0/(rank+1))
                    break
        #assert(len(mrr) == len(test_users))
        return np.mean(mrr)



    def fit(self, X, debug=False):
        self.U = 0.01 * np.random.random_sample((X.shape[0], self.K))
        self.V = 0.01 * np.random.random_sample((X.shape[1], self.K))

        if debug:
            num_train_sample_users = min(data.shape[0], 1000)
            train_sample_users = random.sample(
                range(data.shape[0]), num_train_sample_users)
            #print('train mrr = {0:.4f}'.format(
            #    self.compute_mrr(data, self.U, self.V, train_sample_users)))

        for iter in range(self.max_iters):
            print('Iteration {0}:'.format(iter+1))
            self.update(X, self.U, self.V, self.lmda, self.gamma)
            if debug:
                print('\t- Objective: {0:.4f}'.format(
                    self.objective(data,self.U, self.V, self.lmda)))
                print('\t- Train MRR: {0:.4f}\n'.format(
                    self.compute_mrr(data, self.U, self.V,
                                     train_sample_users)))



    def recommend(self, user, sort=True, top=None):
        predictions = np.dot(self.U[user], self.V.T)

        # Sort predictions
        if sort:
            indices = np.argsort(predictions)[::-1]  # Descending order
        else:
            indices = np.arange(0, len(predictions))

        # Join predictions and indices
        predictions = np.array((indices, predictions[indices])).T

        # Return top-k recommendations
        if top != None:
            return predictions[:top]

        return predictions




if __name__=='__main__':
    data = io.mmread("EP25_UPL5_train.mtx").tocsr()
    testdata = io.mmread("EP25_UPL5_test.mtx").tocsr()

    recommender = CLiMF()
    recommender.fit(data, debug=False)
    prediction = recommender.recommend(user=1, sort=True, top=None)

    print(prediction)
    print('')
    print(prediction[:, 1].T)
    #correct = np.array([4,  1,  3,  1])
    #np.testing.assert_almost_equal(
    #    np.round(np.round(prediction[:, 1])), np.round(correct))