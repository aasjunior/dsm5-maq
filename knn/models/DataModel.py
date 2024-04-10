import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class DataModel:
    def __init__(self):
        self.df = pd.DataFrame()
        self.numeric_cols = []
        self.categorical_cols = []
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder

    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)

    def set_columns(self, numeric_cols, categorical_cols):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

    def normalize_data(self):
        self.df[self.numeric_cols] = self.scaler.fit_transform(self.df[self.numeric_cols])

        for col in self.categorical_cols:
            self.df[col] = self.label_encoder.fit_transform(self.df[col])

        return self.df