import numpy as np
from numpy.linalg import pinv
import pandas as pd

class ERFS:
    def __init__(self, X, Y):
        """
        Initialize the ERFS object with data and labels.
        :param X: a pandas dataframe to represent data, # instances n * # classes d.
        :param Y: a pandas dataframe to represent the label, # instances n * 1.
        """
        # store as X.transpose() to comply with the form in paper.
        [self.n, self.d] = X.shape
        self.c = Y.shape[1]
        self.m = self.n + self.d
        Y = pd.get_dummies(Y.iloc[:, 0]).astype(int)
        self.features = X.index
        X = np.array(X)
        Y = np.array(Y)
        self.X = X.transpose()
        self.Y = Y

    def fit(self, gamma, epoch=15):
        """
        This function implements the Efficient and Robust Feature Selection (REFS) algorithm.

        :param gamma: a number, see paper for more detail.
        """
        # initialize variables
        A = np.hstack((self.X.transpose(), gamma * np.eye(self.n)))
        U = None
        Dt = np.eye(self.m)

        # iterate for certain epochs
        for t in range(epoch):
            part1 = pinv(Dt) @ A.transpose()
            part2 = pinv(A @ pinv(Dt) @ A.transpose())
            U = part1 @ part2 @ self.Y
            Dt = U @ U.transpose()
            Dt = pinv(Dt * np.eye(Dt.shape[0])) * 0.5

        # calculate the weight matrix
        self.W = U[:self.d, :]

        # calculate feature importance
        row_sum = np.sum(np.abs(self.W), axis=1)

        # convert to pandas dataframe
        self.feature_importances = pd.DataFrame(row_sum, index=self.features, columns=['importance'])

        # sort the score in descending order
        self.feature_importance = self.feature_importance.sort_values('importance', ascending=False)


    def k_features(self, k=None):

        copy = self.feature_importance.copy()
        if k is not None and k < copy.shape[0]:
            return copy[1:k].index.tolist()
        else:
            return copy.index.tolist()
