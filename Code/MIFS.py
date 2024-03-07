import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import line_search

class MIFS:
    def __init__(self, data, label):
        """
        Initializes the MIFS object with data and labels.

        :param data: a pandas dataframe to represent data, # instances n * # classes d.
        :param label: a pandas dataframe to represent the label, # instances n * 1.
        :param sqrd_sigma: a floating point to represent kernel width.
        :param p: an integer to represent the number of nearest neighbors.
        """
        [self.n, self.d] = data.shape
        self.k = label.shape[1]
        self.data = np.array(data)
        self.label = np.array(label)

    @staticmethod
    def theta(X, Y, L, W, V, B, alpha, beta, gamma):
        """
        This is the objective function that the algorithm tries to minimize.

        :param X: a numpy matrix that represents the data point.
        :param Y: a numpy matrix that represents the label.
        :param L: a numpy matrix, see paper for more detail.
        :param W: a numpy matrix that represents weights.
        :param V: a numpy matrix that represents latent semantics.
        :param B: a numpy matrix that represents the coefficient of latent semantics.
        :param alpha: a floating point.
        :param beta: a floating point.
        :param gamma: a floating point.

        :return: the objective function value.
        """

        result = ((np.linalg.norm(X @ W - V, 'fro') ** 2 +
                  alpha * (np.linalg.norm(Y - V @ B)**2)) +
                  beta * np.trace(V.tranpose() @ L @ V) +
                  gamma * np.sum(np.linalg.norm(W, axis=1)))

        return result

    def fit(self,
            l,
            alpha=0.4,
            beta=1,
            gamma=0.8,
            epoch=10,
            sqrd_sigma=0.9,
            epislon=0.001,
            p=5):

        # calculate L matrix
        # L = A - S
        distance_matrix = cdist(self.data, self.data, 'euclidean')
        p_nearest_neighbors = np.argsort(distance_matrix)[:, 1:p+1]
        kernal_distance_matrix = np.exp(-1 * distance_matrix / sqrd_sigma)
        S = np.zeros((self.n, self.n))
        for i in range(self.n):
            self.S[i, p_nearest_neighbors[i]] = kernal_distance_matrix[i, p_nearest_neighbors[i]]
        Aii = np.sum(self.S, axis=1)
        A = np.diag(Aii)
        L = A - S

        if l is None:
            l = int(self.k / 2)
        W = np.zeros((self.d, l))
        V = np.zeros((self.n, l))
        B = np.zeros(l, self.k)
        X = self.data
        Y = self.label

        # start computation
        for i in range(epoch):

            # calculate D matrix
            W_sqrd = W @ W.transpose()
            W_sqrd = W_sqrd * np.eye(W_sqrd.shape[0])
            W_sqrd = W_sqrd + np.eye(W_sqrd.shape[0]) * epislon
            D = 2 * np.exp(W_sqrd)

            # calculate derivative
            d_theta_d_W = 2 * (X.transpose() @ (X @ W - V) + gamma * D @ W)
            d_theta_d_V = 2 * ((V - X @ W) + alpha * (V @ B - Y) @ B.transpose() + beta * L @ V)
            d_theta_d_B = 2 * alpha * V.tranpose() @ (V @ B - Y)

            # use Armijo rule to determine stepsizes lambda for W, V, and B
            search_result = line_search(f=lambda w: self.theta(X, Y, L, w, V, B, alpha, beta, gamma),
                                   myfprime=lambda w: 2 * (X.transpose() @ (X @ W - V) + gamma * D @ W),
                                   xk=W,
                                   pk=d_theta_d_W)
            lambda_W = 0.001
            if search_result is not None:
                lambda_W = search_result[0]


        pass