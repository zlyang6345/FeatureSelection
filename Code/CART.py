import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class FeatureSelection:
    data = pd.DataFrame()
    labels = pd.DataFrame()
    target_labels = None
    feature_importance = pd.DataFrame()

    def cart_fit(self):
        clf = DecisionTreeClassifier()
        clf.fit(self.data, self.label)
        self.feature_importance = pd.DataFrame(clf.feature_importances_, index=self.data.columns,
                                               columns=['importance'])
        self.feature_importance = self.feature_importance.sort_values('importance', ascending=False)
        pass

    def k_features(self, k=None):

        copy = self.feature_importance.copy()
        if k is not None and k < copy.shape[0]:
            return copy[1:k].index.tolist()
        else:
            return copy.index.tolist()

    def __init__(self, data, label, target_labels=None):

        self.data = data
        self.label = label

        if target_labels is not None:
            target_labels_set = set(target_labels)
            in_the_set_index = self.labels.iloc[:, 0].apply(lambda x: x in target_labels_set)
            self.labels = self.labels[in_the_set_index]
            self.data = self.data[in_the_set_index]

        self.cart_fit()
