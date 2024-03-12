import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import line_search
import pandas as pd
import matplotlib.pyplot as plt


class MIFS:
    def __init__(self, data, label):
        """
        Initializes the MIFS object with data and labels.

        :param data: A pandas dataframe to represent data, # instances n * # classes d.
        :param label: A pandas dataframe to represent the label, # instances n * 1.
        :param sqrd_sigma: A floating point to represent kernel width.
        :param p: An integer to represent the number of nearest neighbors.
        """
        [self.n, self.d] = data.shape
        self.features = data.columns
        self.data = np.array(data)
        label = pd.get_dummies(label.iloc[:, 0].astype('category')).astype(int)
        self.k = label.shape[1]
        self.label = np.array(label)

    @staticmethod
    def theta(X, Y, L, W, V, B, alpha=0.4, beta=1, gamma=0.8):
        """
        This is the objective function that the algorithm tries to minimize.

        :param X: A numpy matrix that represents the data point.
        :param Y: A numpy matrix that represents the label.
        :param L: A numpy matrix, see paper for more detail.
        :param W: A numpy matrix that represents weights.
        :param V: A numpy matrix that represents latent semantics.
        :param B: A numpy matrix that represents the coefficient of latent semantics.
        :param alpha: A floating point.
        :param beta: A floating point.
        :param gamma: A floating point.

        :return: The objective function value.
        """

        result = ((np.linalg.norm(X @ W - V, 'fro') ** 2 +
                   alpha * (np.linalg.norm(Y - V @ B) ** 2)) +
                  beta * np.trace(V.transpose() @ L @ V) +
                  gamma * np.sum(np.linalg.norm(W, axis=1)))

        return result

    @staticmethod
    def armijo(loss, start, gradient, epochs=10):
        """
        This function implements the Armijo rule.

        :param loss: A loss function.
        :param start: A numpy matrix that represents the starting point.
        :param gradient: A numpy matrix that represents the gradient.
        :param epochs: The maximum number of iterations.
        :return: None or alpha.
        """
        gamma = 0.1
        c = 0.5
        alpha = 1
        d = -1 * gradient
        epoch = 0
        while (loss(start + alpha * d)
               > (loss(start) + c * alpha * np.trace(gradient.transpose() @ d))):
            alpha = gamma * alpha
            epoch += 1
            if epoch > epochs:
                return None
        return alpha

    def fit(self,
            c,
            alpha=0.4,
            beta=1,
            gamma=0.8,
            epoch=200,
            sqrd_sigma=0.9,
            epislon=0.001,
            p=5):

        # calculate L matrix
        # L = A - S
        distance_matrix = cdist(self.data, self.data, 'euclidean')
        p_nearest_neighbors = np.argsort(distance_matrix)[:, 1:p + 1]
        kernal_distance_matrix = np.exp(-1 * distance_matrix / sqrd_sigma)
        S = np.zeros((self.n, self.n))
        for i in range(self.n):
            S[i, p_nearest_neighbors[i]] = kernal_distance_matrix[i, p_nearest_neighbors[i]]
        S = np.sqrt(
            S * S.transpose())  # this step makes sure the affinity matrix is symmetric (i and j should treat each other as their nearest neighbors. Otherwise, should be 0.)
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
        W = np.random.rand(self.d, c)
        V = np.random.rand(self.n, c)
        B = np.random.rand(c, self.k)
        X = self.data
        Y = self.label

        thetas = list()

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

            # use the Armijo rule to determine a stepsize lambda for W, V, and B
            # then update W, V, B
            lbd = 1e-4
            lambda_W = lbd
            lambda_B = lbd
            lambda_V = lbd
            # search lambda_W
            search_result = MIFS.armijo(loss=lambda w: self.theta(X, Y, L, w, V, B),
                                        start=W,
                                        gradient=d_theta_d_W)
            if search_result is not None:
                lambda_W = search_result

            # search lambda_B
            search_result = MIFS.armijo(loss=lambda b: self.theta(X, Y, L, W, V, b),
                                        start=B,
                                        gradient=d_theta_d_B)

            if search_result is not None:
                lambda_B = search_result

            # search lambda_V
            search_result = MIFS.armijo(loss=lambda v: self.theta(X, Y, L, W, v, B),
                                        start=V,
                                        gradient=d_theta_d_V)

            if search_result is not None:
                lambda_V = search_result

            # update W, V, and B
            W = W - lambda_W * d_theta_d_W
            V = V - lambda_V * d_theta_d_V
            B = B - lambda_B * d_theta_d_B

            thetas.append(self.theta(X, Y, L, W, V, B, alpha, beta, gamma))

        # plot the result
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(thetas)
        ax.set_yscale('log')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Theta(Loss, Logarithmic)')
        plt.title(f'With Fixed Stepsize {lbd}')
        plt.show()

        # process the result
        self.W = W
        # calculate feature importance
        row_sum = np.sum(np.abs(self.W * self.W), axis=1)
        # convert to pandas dataframe
        self.feature_importance = pd.DataFrame(row_sum, index=self.features, columns=['importance'])
        # sort the score in descending order
        self.feature_importance = self.feature_importance.sort_values('importance', ascending=False)
