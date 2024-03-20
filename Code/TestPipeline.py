import pandas as pd


class TestPipeline(object):

    def __init__(self):
        self.processed_labels = pd.DataFrame()
        self.data = pd.DataFrame()
        self.label = pd.DataFrame()
        self.feature_importance = pd.DataFrame()
        self.target_labels = set()
        self.k = int()

    def read_preprocess_data(self,
                             target_labels,
                             data_file_path='../Data/murine_spleen_protein_normalized.csv',
                             label_file_path='../Data/cite_cluster_labels.csv', ):
        """
        Read from data and labels given file paths and preprocess the label.

        :param target_labels: A set to represent the target labels.
        :param data_file_path: A string to represent data file address.
        :param label_file_path: A string to represent label file address.
        """

        self.data = pd.read_csv(data_file_path,
                                header=0,
                                index_col=0).astype(float)
        self.label = pd.read_csv(label_file_path,
                                 index_col=0,
                                 header=0).astype(int)

        self.target_labels = target_labels
        self.processed_labels = self.label.applymap(
            lambda x: -1 if x not in self.target_labels else x)
