from orangecontrib.recommendation import *


def test_loss():
    
    # Load data
    data = Orange.data.Table('filmtrust/ratings_small.tab')
    trust = Orange.data.Table('filmtrust/trust_small.tab')

    # Train recommender
    print('*************** BRISMF ***************')
    brismf = BRISMFLearner(num_factors=15, num_iter=10, learning_rate=0.07,
                           lmbda=0.1, verbose=True)
    brismf(data)
    print('')

    print('*************** SVD++ ***************')
    svdpp = SVDPlusPlusLearner(num_factors=2, num_iter=10, learning_rate=0.007,
                               lmbda=0.01, verbose=True)
    svdpp(data)
    print('')

    print('*************** TrustSVD ***************')
    trustsvd = TrustSVDLearner(num_factors=15, num_iter=10, learning_rate=0.007,
                               lmbda=0.01, social_lmbda=0.05, trust=trust,
                               verbose=True)
    trustsvd(data)

if __name__ == "__main__":
    test_loss()