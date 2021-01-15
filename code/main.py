
# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random_forest
import naive_bayes
import knn
import stacking
import linear_model


# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# our code
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":

        filename_train = "wordvec_train.csv"
        filename_test = "wordvec_test.csv"
        with open(os.path.join("..", "data", filename_train), "rb") as f:
            data_train = pd.read_csv(f,header=0)

        with open(os.path.join("..", "data", filename_test), "rb") as f:
            data_test = pd.read_csv(f,header=0)

        X = data_train[data_train.columns[0:200]].values
        y = data_train["Target"].values
        X_test = data_test[data_test.columns[0:200]].values
        y_test = data_test["Target"].values

        model = random_forest.RandomForest(15)
        utils.evaluate_model(model,X,y,X_test,y_test)
        # model.fit(X,y)
        # y_train_pred = model.predict(X)
        # y_test_pred = model.predict(X_test)

        # train_error = np.mean(y != y_train_pred)
        # test_error = np.mean(y_test != y_test_pred)

        # print("The training error is {}".format(train_error))
        # print("The test error is {}".format(test_error))
        

    elif question == "2":
        filename_train = "wordvec_train.csv"
        filename_test = "wordvec_test.csv"
        with open(os.path.join("..", "data", filename_train), "rb") as f:
            data_train = pd.read_csv(f,header=0)

        with open(os.path.join("..", "data", filename_test), "rb") as f:
            data_test = pd.read_csv(f,header=0)

        X = data_train[data_train.columns[0:200]].values
        y = data_train["Target"].values
        X_test = data_test[data_test.columns[0:200]].values
        y_test = data_test["Target"].values

        model = naive_bayes.NaiveBayes()
        utils.evaluate_model(model, X, y, X_test, y_test)
    
    elif question == "3":
        filename_train = "wordvec_train.csv"
        filename_test = "wordvec_test.csv"
        with open(os.path.join("..", "data", filename_train), "rb") as f:
            data_train = pd.read_csv(f,header=0)

        with open(os.path.join("..", "data", filename_test), "rb") as f:
            data_test = pd.read_csv(f,header=0)

        X = data_train[data_train.columns[0:200]].values
        y = data_train["Target"].values
        X_test = data_test[data_test.columns[0:200]].values
        y_test = data_test["Target"].values

        model = KNeighborsClassifier(3, metric="cosine")
        utils.evaluate_model(model,X,y,X_test,y_test)

    elif question == "4":
        filename_train = "wordvec_train.csv"
        filename_test = "wordvec_test.csv"
        with open(os.path.join("..", "data", filename_train), "rb") as f:
            data_train = pd.read_csv(f,header=0)

        with open(os.path.join("..", "data", filename_test), "rb") as f:
            data_test = pd.read_csv(f,header=0)

        X = data_train[data_train.columns[0:200]].values
        y = data_train["Target"].values
        X_test = data_test[data_test.columns[0:200]].values
        y_test = data_test["Target"].values

        model = stacking.Stacking()
        utils.evaluate_model(model,X,y,X_test,y_test)

    elif question == "5":
        filename = "phase1_training_data.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            data = pd.read_csv(f,header=0)
        
        allData = data.values
        canadian_death = allData[allData["country_id"] == "CA", 3]
        model = linear_model.AutoregreSimpleFeature()
        model.fit()




