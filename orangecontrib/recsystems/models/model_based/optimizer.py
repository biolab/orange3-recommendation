import theano
import theano.tensor as T
import numpy as np
import time
import sys

def NMFN(X,r,iterations,H=None,W=None):
    rng = np.random
    n = np.size(X,0)
    m = np.size(X,1)
    if(H is None):
        H = rng.random((r,m)).astype(theano.config.floatX)
    if(W is None):
        W = rng.random((n,r)).astype(theano.config.floatX)

    for i in range(0,iterations):
        #print np.linalg.norm(X-np.dot(W,H))
        H = H*(np.dot(W.T,X)/np.dot(np.dot(W.T,W),H))
        W = W*(np.dot(X,H.T)/np.dot(np.dot(W,H),H.T))

    return W,H


def NMF(X,r,iterations,H=None,W=None):
    rng = np.random
    n = np.size(X,0)
    m = np.size(X,1)
    if(H is None):
        H = rng.random((r,m)).astype(theano.config.floatX)
    if(W is None):
        W = rng.random((n,r)).astype(theano.config.floatX)

    tX = theano.shared(X.astype(theano.config.floatX),name="X")
    tH = theano.shared(H,name="H")
    tW = theano.shared(W,name="W")
    tE = T.sqrt(((tX-T.dot(tW,tH))**2).sum())

    trainH = theano.function(
            inputs=[],
            outputs=[],
            updates={tH:tH*((T.dot(tW.T,tX))/(T.dot(T.dot(tW.T,tW),tH)))},
            name="trainH")
    trainW = theano.function(
            inputs=[],
            outputs=[],
            updates={tW:tW*((T.dot(tX,tH.T))/(T.dot(tW,T.dot(tH,tH.T))))},
            name="trainW")

    for i in range(0,iterations):
        #print np.linalg.norm(X-np.dot(tW.get_value(),tH.get_value()))
        trainH();
        trainW();

    return tW.get_value(),tH.get_value()

if __name__=="__main__":
    print("USAGE : NMF.py <matrixDim> <latentFactors> <iter>")
    n = int(sys.argv[1])
    r = int(sys.argv[2])
    it = int(sys.argv[3])
    rng = np.random
    Hi = rng.random((r,n)).astype(theano.config.floatX)
    Wi = rng.random((n,r)).astype(theano.config.floatX)
    X = rng.random((n,n)).astype(theano.config.floatX)
    t0 = time.time()
    W,H = NMF(X,r,it,Hi,Wi)
    t1 = time.time()
    print("Time taken by Theano : %s", str(t1-t0))
    print(" --- ")
    t0 = time.time()
    W,H = NMFN(X,r,it,Hi,Wi)
    t1 = time.time()
    print("Time taken by CPU : %s", str(t1-t0))