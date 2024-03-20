import numpy as np
from numpy.linalg import pinv
import pandas as pd


class ERFS:
    def __init__(self, data, label):
        """
        Initialize the ERFS object with data and labels.
        :param data: A pandas dataframe to represent data, # instances n * # classes d.
        :param label: A pandas dataframe to represent the label, # instances n * 1.
        """
        self.feature_importance = None
        self.W = None

        # store as X.transpose() to comply with the form in paper.
        X = data
        Y = label
        [self.n, self.d] = X.shape
        self.c = Y.shape[1]
        self.m = self.n + self.d
        Y = pd.get_dummies(Y.iloc[:, 0].astype('category')).astype(int)
        self.features = X.columns
        X = np.array(X)
        Y = np.array(Y)
        self.X = X.transpose()
        self.Y = Y

    @staticmethod
    def loss(X, W, Y, gamma):
        """
        This is the objective function that ERFS algorithm tries to minimize.

        :param X: A numpy array that represents the data.
        :param W: A numpy array that represents the weight.
        :param Y: A numpy array that represents the label.
        :param gamma:
        :return:
        """
        first_part = X.transpose() @ W - Y
        l_2_1_norm = lambda X: np.sum(np.linalg.norm(X, axis=1))
        return l_2_1_norm(first_part) + gamma * l_2_1_norm(W)

    def fit(self, gamma, epoch=15, non_zero=True, sigma=0.001):
        """
        This function implements the Efficient and Robust Feature Selection (ERFS) algorithm.

        :param gamma: A number, see paper for more detail.
        :param epoch: An integer, the number of iterations.
        :param non_zero: A boolean, whether to keep zero importance-score features.
        :param sigma: A float number, a regularizer parameter.
        :
        """
        # initialize variables
        A = np.hstack((self.X.transpose(), gamma * np.eye(self.n)))
        U = None
        Dt = np.eye(self.m)
        losses = list()

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
            losses.append(ERFS.loss(self.X, U[:self.d, :], self.Y, gamma=gamma))

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

        return losses

    def k_features(self, k=None):
        """
        This function should be executed after fit.
        The function will return k important features.

        :param k: An integer to specify the desired number of features.
        :return: A list that includes k important features.
        """
        copy = self.feature_importance.copy()
        if k is not None and k < copy.shape[0]:
            return copy[0:k].index.tolist()
        else:
            return copy.index.tolist()
