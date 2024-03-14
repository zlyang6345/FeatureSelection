import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
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
    def armijo(loss, start, gradient, epochs=10, alpha=1, gamma=0.1):
        """
        This function implements the Armijo rule.

        :param loss: A loss function.
        :param start: A numpy matrix that represents the starting point.
        :param gradient: A numpy matrix that represents the gradient.
        :param epochs: The maximum number of iterations.
        :return: None or alpha.
        """
        c = 0.8
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
            epsilon=0.001,
            p=5):
        """
        Implement the MIFS algorithm, according to the paper.
        https://www.ijcai.org/Proceedings/16/Papers/233.pdf

        :param c: An integer to represent the dimension of latent space.
        :param alpha: A float number, see original paper for more detail.  
        :param beta: A float number, see original paper for more detail. 
        :param gamma: A float number, see original paper for more detail. 
        :param epoch: An integer to represent the maximum number of iterations.
        :param sqrd_sigma: A float to represent the kernel width. 
        :param epsilon: A minimal float number to prevent singularity. 
        :param p: An integer number to represent the number of closest neighbors. 
        return: A list that contains loss as function iterates. 
        """
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

        # lambda_Ws = list()
        # lambda_Vs = list()
        # lambda_Bs = list()
        thetas = [self.theta(X, Y, L, W, V, B)]

        # start computation
        for i in range(epoch):

            # calculate D matrix
            W_sqrd = W @ W.transpose()
            W_sqrd = W_sqrd * np.eye(W_sqrd.shape[0])
            W_sqrd = W_sqrd + np.eye(W_sqrd.shape[0]) * epsilon
            D = 2 * np.sqrt(W_sqrd)
            D = np.linalg.pinv(D)

            # calculate derivative
            d_theta_d_W = 2 * (X.transpose() @ (X @ W - V) + gamma * D @ W)
            d_theta_d_V = 2 * ((V - X @ W) + alpha * (V @ B - Y) @ B.transpose() + beta * L @ V)
            d_theta_d_B = 2 * alpha * V.transpose() @ (V @ B - Y)

            # use the Armijo rule to determine a stepsize lambda for W, V, and B
            # then update W, V, B
            # search lambda_W
            search_result = MIFS.armijo(loss=lambda w: self.theta(X, Y, L, w, V, B),
                                        start=W,
                                        gradient=d_theta_d_W,
                                        gamma=0.1)
            if search_result is not None:
                lambda_W = search_result
                W = W - lambda_W * d_theta_d_W
            # lambda_Ws.append(search_result)

            # search lambda_B
            search_result = MIFS.armijo(loss=lambda b: self.theta(X, Y, L, W, V, b),
                                        start=B,
                                        gradient=d_theta_d_B,
                                        gamma=0.1)
            # lambda_Bs.append(search_result)
            if search_result is not None:
                lambda_B = search_result
                B = B - lambda_B * d_theta_d_B

            # search lambda_V
            search_result = MIFS.armijo(loss=lambda v: self.theta(X, Y, L, W, v, B),
                                        start=V,
                                        gradient=d_theta_d_V,
                                        gamma=0.5)

            if search_result is not None:
                lambda_V = search_result
                V = V - lambda_V * d_theta_d_V
            # lambda_Vs.append(search_result)

            thetas.append(self.theta(X, Y, L, W, V, B, alpha, beta, gamma))

        # plot the result
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # ax.plot(thetas)
        # ax.set_yscale('log')
        # ax.set_xlabel('Epochs')
        # ax.set_ylabel('Theta(Loss, Logarithmic)')
        # plt.show()

        # process the result
        self.W = W
        # calculate feature importance
        row_sum = np.sum(np.abs(self.W * self.W), axis=1)
        # convert to pandas dataframe
        self.feature_importance = pd.DataFrame(row_sum, index=self.features, columns=['importance'])
        # sort the score in descending order
        self.feature_importance = self.feature_importance.sort_values('importance', ascending=False)

        return thetas

    def quadratic_fit(self,
                      c,
                      sqrd_sigma=0.9,
                      epoch=200,
                      epsilon=0.001,
                      alpha=0.4,
                      beta=1,
                      gamma=0.8,
                      p=5, ):
        """
        This is a modified version of the original MIFS algorithm.
        It employs a quasi-Newton method to do the gradient descent.

        :param c: An integer to represent the dimension of latent space. 
        :param sqrd_sigma: A float to represent the kernel width. 
        :param epoch: An integer to represent the maximum number of iterations. 
        :param epsilon: A minimal float number to prevent singularity. 
        :param alpha: A float number, see original paper for more detail.  
        :param beta: A float number, see original paper for more detail. 
        :param gamma: A float number, see original paper for more detail. 
        :param p: An integer number to represent the number of closest neighbors. 
        :return: A list that contains loss as function iterates. 
        """
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

        thetas = [self.theta(X, Y, L, W, V, B)]

        for i in range(epoch):

            # calculate D matrix
            W_sqrd = W @ W.transpose()
            W_sqrd = W_sqrd * np.eye(W_sqrd.shape[0])
            W_sqrd = W_sqrd + np.eye(W_sqrd.shape[0]) * epsilon
            D = 2 * np.sqrt(W_sqrd)
            D = np.linalg.pinv(D)

            # calculate derivative
            d_theta_d_W = 2 * (X.transpose() @ (X @ W - V) + gamma * D @ W)
            d_theta_d_V = 2 * ((V - X @ W) + alpha * (V @ B - Y) @ B.transpose() + beta * L @ V)
            d_theta_d_B = 2 * alpha * V.transpose() @ (V @ B - Y)

            # update W
            res_W = minimize(fun=lambda w: self.theta(X, Y, L, w.reshape(W.shape), V, B),
                     x0=W.flatten(),
                     method='L-BFGS-B',
                     jac=lambda w: (2 * (X.transpose() @ (X @ w.reshape(W.shape) - V) + gamma * D @ w.reshape(W.shape))).flatten())

            if res_W.success:
                W = res_W.x.reshape(W.shape)

            # update V
            res_V = minimize(fun=lambda v: self.theta(X, Y, L, W, v.reshape(V.shape), B),
                             x0=V.flatten(),
                             method='L-BFGS-B',
                             jac=lambda v: (2 * ((v.reshape(V.shape) - X @ W) + alpha * (v.reshape(V.shape) @ B - Y) @ B.transpose() + beta * L @ v.reshape(V.shape))).flatten())
            if res_V.success:
                V = res_V.x.reshape(V.shape)

            # update B
            res_B = minimize(fun=lambda b: self.theta(X, Y, L, W, V, b.reshape(B.shape)),
                             x0=B.flatten(),
                             method='L-BFGS-B',
                             jac=lambda b: (2 * alpha * V.transpose() @ (V @ b.reshape(B.shape) - Y)).flatten())
            if res_B.success:
                B = res_B.x.reshape(B.shape)

            thetas.append(self.theta(X, Y, L, W, V, B, alpha, beta, gamma))

        return thetas
