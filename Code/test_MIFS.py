from unittest import TestCase

import matplotlib.pyplot as plt
import pandas as pd
from Code.MIFS import MIFS

class TestMIFS(TestCase):
    def test_fit(self):
        """
        A simple test to see whether the algorithm will work
        and return the specified number of important features.
        """
        data = pd.read_csv('../Data/murine_spleen_protein_normalized.csv', header=0, index_col=0).astype(float)
        label = pd.read_csv('../Data/cite_cluster_labels.csv', index_col=0).astype(int)

        # for test speed
        # only take 100
        data = data.iloc[:2000, :]
        # normalize each row
        # Define a lambda function for min-max normalization
        min_max_normalize = lambda x: (x - x.min()) / (x.max() - x.min())
        # Apply the function to each row
        normalized_data = data.apply(min_max_normalize, axis=1)
        label = label.iloc[:2000, :]

        mifs = MIFS(normalized_data, label)
        mifs.fit(c=None)

    def test_quadratic_fit(self):
        """
        A simple test to see whether the algorithm will work and return the specified number of important features.
        """
        data = pd.read_csv('../Data/murine_spleen_protein_normalized.csv', header=0, index_col=0).astype(float)
        label = pd.read_csv('../Data/cite_cluster_labels.csv', index_col=0).astype(int)

        # for test speed
        # only take 100
        data = data.iloc[:2000, :]
        # normalize each row
        # Define a lambda function for min-max normalization
        min_max_normalize = lambda x: (x - x.min()) / (x.max() - x.min())
        # Apply the function to each row
        normalized_data = data.apply(min_max_normalize, axis=1)
        label = label.iloc[:2000, :]

        mifs = MIFS(normalized_data.copy(), label.copy())
        quadratic_thetas = mifs.quadratic_fit(c=None, epoch=10)

        mifs = MIFS(normalized_data.copy(), label.copy())
        regular_thetas = mifs.fit(c=None, epoch=10)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(quadratic_thetas, label='Improved MIFS')
        ax.plot(regular_thetas, label='Original MIFS')
        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Thetas (Loss, Logarithmic)')
        ax.set_title('Comparison between Modified and Original MIFS')
        plt.show()


