from unittest import TestCase
import pandas as pd
from Code.CART import FeatureSelection


class TestCART(TestCase):
    def test_k_features(self):
        data = pd.read_csv('../Data/murine_spleen_protein_normalized.csv', header=0, index_col=0).astype(float)
        label = pd.read_csv('../Data/cite_cluster_labels.csv', index_col=0).astype(int)

        fs = FeatureSelection(data, label)
        k_features = fs.k_features()
        assert k_features is not None
        assert len(k_features) == data.shape[1]

    def test_effect(self):
        data = pd.read_csv('../Data/murine_spleen_protein_normalized.csv', header=0, index_col=0).astype(float)
        label = pd.read_csv('../Data/cite_cluster_labels.csv', index_col=0).astype(int)

        # to implement
