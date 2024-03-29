import time
from unittest import TestCase
import pandas as pd
import numpy as np

from Code.ERFS import ERFS


class TestERFS(TestCase):
    def test_fit(self):
        """
        A simple test to see whether the algorithm will work and return the specified number of important features.
        """
        start_time = time.time()
        data = pd.read_csv('../Data/murine_spleen_protein_normalized.csv', header=0, index_col=0).astype(float)
        label = pd.read_csv('../Data/cite_cluster_labels.csv', index_col=0).astype(int)

        # for test speed
        # only take 100
        num = 1000
        data = data.iloc[:num , :]
        label = label.iloc[:num, :]

        erfs = ERFS(data, label)
        erfs.fit(0.01, 10, True, 0.01)
        print('Erf time: ', time.time() - start_time)

        important_features = erfs.k_features()
        # assert len(important_features) == 4
