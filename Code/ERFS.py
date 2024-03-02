import numpy as np
from numpy.linalg import pinv
import pandas as pd


class ERFS:
    def __init__(self, data, label):
        """
        Initialize the ERFS object with data and labels.
        :param data: a pandas dataframe to represent data, # instances n * # classes d.
        :param label: a pandas dataframe to represent the label, # instances n * 1.
        """
        # store as X.transpose() to comply with the form in paper.
        X = data
        Y = label
        [self.n, self.d] = X.shape
        self.c = Y.shape[1]
        self.m = self.n + self.d
        Y = pd.get_dummies(Y.iloc[:, 0]).astype(int)
        self.features = X.columns
        X = np.array(X)
        Y = np.array(Y)
        self.X = X.transpose()
        self.Y = Y

    def fit(self, gamma, epoch=15, non_zero=True, sigma=0.001):
        """
        This function implements the Efficient and Robust Feature Selection (REFS) algorithm.

        :param gamma: a number, see paper for more detail.
        :param epoch: an integer, the number of iterations.
        :param non_zero: a boolean, whether to keep zero importance-score features.
        :param sigma: a float number, a regularizer parameter.
        :
        """
        # initialize variables
        A = np.hstack((self.X.transpose(), gamma * np.eye(self.n)))
        U = None
        Dt = np.eye(self.m)

        # iterate for certain epochs
        for t in range(epoch):
            # use pinv instead of inv for stability
            part1 = pinv(Dt) @ A.transpose()
            part2 = pinv(A @ pinv(Dt) @ A.transpose())
            U = part1 @ part2 @ self.Y
            Dt = U @ U.transpose()
            diag = Dt * np.eye(Dt.shape[0])
            # a recommended way for singular matrix
            # sometimes the diagonal element will be 0. So add sigma.
            # see the original paper for more detail.
            diag = diag + np.eye(diag.shape[0]) * sigma
            diag = np.sqrt(diag)
            Dt = pinv(diag) * 0.5

        # calculate the weight matrix
        self.W = U[:self.d, :]

        # calculate feature importance
        row_sum = np.sum(np.abs(self.W), axis=1)

        # convert to pandas dataframe
        self.feature_importance = pd.DataFrame(row_sum, index=self.features, columns=['importance'])

        # sort the score in descending order
        self.feature_importance = self.feature_importance.sort_values('importance', ascending=False)

        # will only select non-zero elements
        if non_zero:
            non_zero_indices = (self.feature_importance.iloc[:, 0] != 0)
            self.feature_importance = self.feature_importance[non_zero_indices]
        pass

    def k_features(self, k=None):
        """
        This function should be executed after fit.
        The function will return k important features.
        :param k: an integer to specify the descired number of features.
        :return: a list that includes k important features.
        """
        copy = self.feature_importance.copy()
        if k is not None and k < copy.shape[0]:
            return copy[0:k].index.tolist()
        else:
            return copy.index.tolist()
