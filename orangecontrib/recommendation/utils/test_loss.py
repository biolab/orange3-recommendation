import Orange
from orangecontrib.recommendation import *

def test_loss():
    # Load data
    data = Orange.data.Table('filmtrust/ratings_small.tab')
    trust = Orange.data.Table('filmtrust/trust_small.tab')


    # Train recommender
    print('BRISMF **************************')
    brismf = BRISMFLearner(K=15, steps=10, alpha=0.07, beta=0.1, verbose=True)
    recommender = brismf(data)

    print('SVD++ **************************')
    svdpp = SVDPlusPlusLearner(K=2, steps=10, alpha=0.007, beta=0.01,
                               verbose=True)
    recommender = svdpp(data)

    print('TrustSVD **************************')
    trustsvd = TrustSVDLearner(K=15, steps=10, alpha=0.007, beta=0.01,
                               beta_trust=0.05, trust=trust, verbose=True)
    recommender = trustsvd(data)

if __name__ == "__main__":
    test_loss()