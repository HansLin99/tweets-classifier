"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y

    def predict(self, Xtest):
        X = self.X
        y = self.y
        n = X.shape[0]
        t = Xtest.shape[0]
        k = min(self.k, n)

        # Compute cosine_distance distances between X and Xtest
        dist2 = cosine_distance(X, Xtest)

        # yhat is a vector of size t with integer elements
        yhat = np.ones(t, dtype=np.uint8)
        for i in range(t):
            # sort the distances to other points
            inds = np.argsort(dist2[:,i])

            # compute mode of k closest training pts
            yhat[i] = stats.mode(y[inds[:k]])[0][0]

        return yhat
    
    
def cosine_distance(X1,X2):
    # Compute norms
    norms1 = np.squeeze(np.sum(np.abs(X1)**2, axis=-1)**(1./2))
    norms2 = np.squeeze(np.sum(np.abs(X2)**2, axis=-1)**(1./2))
    # Construct norms matrix
    norms = np.outer(norms1, norms2)
    # Compute dot product
    dotproduct = X1@X2.T
    return np.ones((X1.shape[0],X2.shape[0]))-dotproduct/norms
