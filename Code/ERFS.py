import numpy as np
from numpy.linalg import inv

class ERFS():
    def __init__(self, X, Y):
        """
        Initialize the ERFS object with data and labels.
        :param X: a numpy array, n * d.
        :param Y: a numpy array, n * c
        """
        # store as X.transpose() to comply with the form in paper.
        [self.n, self.d] = X.shape
        self.c = Y.shape[1]
        self.m = self.n + self.d
        self.X = X.transpose()
        self.Y = Y

    def cart_fit(self, gamma, epoch):
        """
        This function implements the Efficient and Robust Feature Selection (REFS) algorithm.

        :param gamma: a number, see paper for more detail.
        """
        A = np.vstack((self.X.transpose(), gamma * np.eye(self.n)))
        U = None
        Dt = np.eye(self.m)
        for t in range(epoch):
            part1 = inv(Dt) @ A.transpose()
            part2 = inv(A @ inv(Dt) @ A.transpose())
            U = part1 @ part2 @ self.Y
            Dt = U @ U.transpose()
            Dt = inv(Dt * np.eye(Dt.shape[0])) * 0.5
        self.W = U[self.d, :]

