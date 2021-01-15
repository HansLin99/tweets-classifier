import numpy as np

class NaiveBayes:

    def __init__(self):
        return

    def fit(self, X, y):
        N, D = X.shape

        y = y.astype(int, copy=False)
        count = np.bincount(y, minlength=2)
        X_trump = X[y==1, :]
        n_1 = X_trump.shape[0]
        X_biden = X[y==0, :]
        n_0 = X_biden.shape[0]
        p_y = count/np.sum(count)

        # Compute the means and variances
        means = np.ones((D, 2))
        variance = np.ones((D, 2))
        for d in range(D):
            means[d, 0] = np.sum(X_biden[:,d])/n_0
            means[d, 1] = np.sum(X_trump[:,d])/n_1
            variance[d, 0] = np.sum(np.square(means[d, 0]-X_biden[:,d]))/n_0 
            variance[d, 1] = np.sum(np.square(means[d,1]-X_trump[:,d]))/n_1

        self.means = means
        self.variance = variance
        self.p_y_log = np.log(p_y)
        
    def predict(self, X):
        N, D = X.shape

        means = self.means
        variance = self.variance
        p_y_log = self.p_y_log
        

        y_pred = np.zeros(N)
        for n in range(N):
            probs = p_y_log.copy() # initialize with the p(y) terms
            for d in range(D):
                # Compute p(y|x) = p(x|y)p(y) in log based
                probs[0] += -np.sum(0.5*np.square(X[n,d]-means[d,0])/variance[d,0]+np.log(np.sqrt(variance[d,0]*2*np.pi)))
                probs[1] += -np.sum(0.5*np.square(X[n,d]-means[d,1])/variance[d,1]+np.log(np.sqrt(variance[d,1]*2*np.pi)))
            
            y_pred[n] = np.argmax(probs)
            

        return y_pred
