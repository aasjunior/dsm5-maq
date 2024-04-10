import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Transforma dados do arquivo em um DataFrame
dados = pd.read_csv("db/iris.data")
# print(dados)

# Remove dados faltantes
dados.dropna(inplace=True)

# Separação dos atributos (X) com a classe (Y)
X = np.array(dados.iloc[:, 0:3]) # Coluna 0 até a coluna 3
Y = np.array(dados['class']) # O atributo referente a coluna da classe chama-se 'class'

# Separação da base de treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, train_size=0.7)

# Quantidade de vizinhos mais próximos
neighboor = 7

# Escolha do algoritmo KNN e envio do valorde k
knn = KNeighborsClassifier(neighboor)

# Etapa de treinamento é necessário fornecr a base de treinamento (X e Y) 
knn.fit(X_train, Y_train)

# Iniciar etapa de teste
previsoes = knn.predict(X_test)

# Medida de desempenho do modelo: acurácia
acuracia = accuracy_score(Y_test, previsoes) * 100

print("A acurácia foi %.2f%%" % acuracia)

