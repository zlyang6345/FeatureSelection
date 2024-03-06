from unittest import TestCase
import pandas as pd
from Code.MIFS import MIFS

class TestMIFS(TestCase):
    def test_fit(self):
        """
        A simple test to see whether the algorithm will work and return the specified number of important features.
        """
        data = pd.read_csv('../Data/murine_spleen_protein_normalized.csv', header=0, index_col=0).astype(float)
        label = pd.read_csv('../Data/cite_cluster_labels.csv', index_col=0).astype(int)

        # for test speed
        # only take 100
        data = data.iloc[:1000, :]
        label = label.iloc[:1000, :]

        mifs = MIFS(data, label)
