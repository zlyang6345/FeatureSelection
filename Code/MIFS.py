import numpy as np

class MIFS:
    def __init__(self, data, label):
        """
        Initializes the MIFS object with data and labels.

        :param data: a pandas dataframe to represent data, # instances n * # classes d.
        :param label: a pandas dataframe to represent the label, # instances n * 1.
        """
        self.data = data
        self.label = label

    def fit(self, alpha, beta, gamma):

        pass