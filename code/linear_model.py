import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self,X,y,z):
        self.w = solve(X.T@z@X, X.T@z@y)

class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w.flatten(), lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):

        ''' MODIFY THIS CODE '''
        # Calculate the function value
        f = np.sum(np.log(np.exp(X @ w - y)+np.exp(y- X@w)))

        # Calculate the gradient value
        g = X.T@((np.exp(X @ w - y)-np.exp(y- X@w))/(np.exp(X @ w - y)+np.exp(y- X@w)))

        return (f,g)


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        Z = np.concatenate((X, np.ones([X.shape[0], 1])), axis=1)
        self.w = solve(Z.T@Z, Z.T@y)

    def predict(self, X):
        Z = np.concatenate((X, np.ones([X.shape[0], 1])), axis=1)
        return Z@self.w

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        Z = self.__polyBasis(X)
        self.w = solve(Z.T@Z, Z.T@y)
            

    def predict(self, X):
        Z = self.__polyBasis(X)
        return Z@self.w

    def __polyBasis(self, X):
        n = X.shape[0]
        Z = np.ones([n,1])
        for po in range(self.p+1):
            Z = np.concatenate((Z, X**(po+1)),axis=1)
        
        return Z

# Simple one feature time series autoregressive model
class AutoregreSimpleFeature:
    def __init__(self, k):
        self.k = k

    def fit(self,X,y):
        Z = self.__buildAuto(X)
        self.w = solve(Z.T@Z, Z.T@y)
            

    def predict(self, X):
        Z = self.__buildAuto(X)
        return Z@self.w

    def __buildAuto(self, X):
        n = X.shape[0]
        Z = np.ones([n,1])
        for po in range(self.k+1):
            Z = np.concatenate((Z, X**(po+1)),axis=1)
        
        return Z

    def featuresSelection(self, lam):
        pass