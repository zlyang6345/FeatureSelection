from unittest import TestCase
import pandas as pd
from Code.CART import CART


class TestCART(TestCase):
    def test_k_features(self):
        data = pd.read_csv('../Data/murine_spleen_protein_normalized.csv', header=0, index_col=0).astype(float)
        label = pd.read_csv('../Data/cite_cluster_labels.csv', index_col=0).astype(int)

        cart = CART(data, label)
        cart.fit(ccp_alpha=0.1)

        print(cart.feature_importance)

        k_features = cart.k_features()
        assert k_features is not None
        assert len(k_features) == data.shape[1]
