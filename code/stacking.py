import numpy as np
import utils
from random_forest import RandomForest
from random_forest import DecisionTree
from random_forest import DecisionStumpGiniIndex
from random_forest import DecisionStumpErrorRate
from knn import KNN
from naive_bayes import NaiveBayes

class Stacking():

    def __init__(self, number_trees=15, max_depth=np.inf, k_knn=3):
        self.number_trees = number_trees
        self.max_depth = max_depth
        self.k_knn = k_knn

    def fit(self, X, y):
        # model1 = RandomForest(15)
        # model2 = KNN(3)
        # model3 = NaiveBayes()
        # model1.fit(X,y)
        # model2.fit(X,y)
        # model3.fit(X,y)
        # y_pred1 = model1.predict(X)
        # y_pred2 = model2.predict(X)
        # y_pred3 = model3.predict(X)
        # X_new = np.column_stack((y_pred1,y_pred2,y_pred3))
        self.model = DecisionTree(5)
        self.model.fit(X, y.astype(int, copy=False))

    def predict(self, X):
        return self.model.predict(X)
