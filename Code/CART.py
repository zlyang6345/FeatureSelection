import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class CART:
    data = pd.DataFrame()
    labels = pd.DataFrame()
    target_labels = None
    feature_importance = pd.DataFrame()

    def fit(self):
        clf = DecisionTreeClassifier()
        clf.fit(self.data, self.label)
        self.feature_importance = pd.DataFrame(clf.feature_importances_,
                                               index=self.data.columns,
                                               columns=['importance'])
        self.feature_importance = self.feature_importance.sort_values('importance',
                                                                      ascending=False)

    def k_features(self, k=None):

        copy = self.feature_importance.copy()
        if k is not None and k < copy.shape[0]:
            return copy[0:k].index.tolist()
        else:
            return copy.index.tolist()

    def __init__(self, data, label):

        self.data = data
        self.label = label
