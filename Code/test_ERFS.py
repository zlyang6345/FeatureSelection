from unittest import TestCase
import pandas as pd

class TestERFS(TestCase):
    def test_cart_fit(self):
        data = pd.read_csv('../Data/murine_spleen_protein_normalized.csv', header=0, index_col=0).astype(float)
        label = pd.read_csv('../Data/cite_cluster_labels.csv', index_col=0).astype(int)

