import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import line_search
import pandas as pd

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
        self.data = np.array(data)
        label = pd.get_dummies(label.iloc[:, 0].astype('category')).astype(int)
        self.k = label.shape[1]
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
                  beta * np.trace(V.transpose() @ L @ V) +
                  gamma * np.sum(np.linalg.norm(W, axis=1)))

        return result

    def fit(self,
            c,
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
            S[i, p_nearest_neighbors[i]] = kernal_distance_matrix[i, p_nearest_neighbors[i]]
        Aii = np.sum(S, axis=1)
        A = np.diag(Aii)
        L = A - S

        if c is None:
            # The latent dimension is not specified.
            if self.k == 1:
                # single label
                c = 1
            else:
                # multiple label
                c = int(2 * self.k / 3)
        W = np.zeros((self.d, c))
        V = np.zeros((self.n, c))
        B = np.zeros((c, self.k))
        X = self.data
        Y = self.label

        # start computation
        for i in range(epoch):

            # calculate D matrix
            W_sqrd = W @ W.transpose()
            W_sqrd = W_sqrd * np.eye(W_sqrd.shape[0])
            W_sqrd = W_sqrd + np.eye(W_sqrd.shape[0]) * epislon
            D = 2 * np.sqrt(W_sqrd)
            D = np.linalg.pinv(D)

            # calculate derivative
            d_theta_d_W = 2 * (X.transpose() @ (X @ W - V) + gamma * D @ W)
            d_theta_d_V = 2 * ((V - X @ W) + alpha * (V @ B - Y) @ B.transpose() + beta * L @ V)
            d_theta_d_B = 2 * alpha * V.transpose() @ (V @ B - Y)

            # use Armijo rule to determine stepsizes lambda for W, V, and B
            # then update W, V, B

            # search lambda_W
            search_result = line_search(f=lambda w: self.theta(X, Y, L, w.reshape(W.shape[0], -1), V, B, alpha, beta, gamma),
                                   myfprime=lambda w: (2 * (X.transpose() @ (X @ w.reshape(W.shape[0], -1) - V) + gamma * D @ w.reshape(W.shape[0], -1))).flatten(),
                                   xk=W.flatten(),
                                   pk=d_theta_d_W.flatten())
            lambda_W = 0.001
            if search_result is not None:
                lambda_W = search_result[0]

            # search lambda_B
            search_result = line_search(f=lambda b: self.theta(X, Y, L, W, V, b, alpha, beta, gamma),
                                   myfprime=lambda b: 2 * alpha * V.tranpose() @ (V @ b - Y),
                                   xk=B,
                                   pk=d_theta_d_B)

            lambda_B = 0.001
            if search_result is None:
                lambda_B = search_result[0]

            # search lambda_V
            search_result = line_search(f=lambda v: self.theta(X, Y, L, W, v, B, alpha, beta, gamma),
                                   myfprime=lambda v: 2 * ((v - X @ W) + alpha * (v @ B - Y) @ B.transpose() + beta * L @ v),
                                   xk=V,
                                   pk=d_theta_d_V)

            lambda_V = 0.001
            if search_result is None:
                lambda_V = search_result[0]

            # update W, V, and B
            W = W - lambda_W * d_theta_d_W
            V = V - lambda_V * d_theta_d_V
            B = B - lambda_B * d_theta_d_B

        # process the result
        self.W = W
        # calculate feature importance
        row_sum = np.sum(np.abs(self.W * self.W), axis=1)
        # convert to pandas dataframe
        self.feature_importance = pd.DataFrame(row_sum, index=self.features, columns=['importance'])
        # sort the score in descending order
        self.feature_importance = self.feature_importance.sort_values('importance', ascending=False)
