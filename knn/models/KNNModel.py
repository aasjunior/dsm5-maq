import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class KNNModel:
    def __init__(self, data_model, k):
        self.data_model = data_model
        self.k = k
        self.knn = KNeighborsClassifier(self.k)
        
    def train(self, X_train, Y_train):
        self.knn.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.knn.predict(X_test)
    
    def evaluate(self, Y_test, predictions):
        accuracy = accuracy_score(Y_test, predictions) * 100
        return "A acurácia foi de %.2f%%" % accuracy
    
    def train_and_evaluate(self, test_size=0.3, train_size=0.7):
        X = np.array(self.data_model.data.iloc[:, 0:len(self.data_model.numeric_cols)]) 
        Y = np.array(self.data_model.data[self.data_model.categorical_cols[0]]) 
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, train_size=train_size)
        
        self.knn.fit(X_train, Y_train)

        predictions = self.knn.predict(X_test)
        accuracy = accuracy_score(Y_test, predictions) * 100
        print("A acurácia foi %.2f%%" % accuracy)