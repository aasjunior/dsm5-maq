import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class DataModel:
    def __init__(self, file_path, numeric_cols, categorical_cols):
        self.file_path = file_path
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder
        self.load_data()
        
    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def set_columns(self, numeric_cols, categorical_cols):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

    def normalize_data(self):
        #self.data[self.numeric_cols] = self.scaler.fit_transform(self.data[self.numeric_cols])

        #for col in self.categorical_cols:
        #    self.data[col] = self.label_encoder.fit_transform(self.data[col].values)

        return self.data.dropna(inplace=True)