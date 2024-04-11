# Algoritmo KNN (Implementação em Python)

* Bibliotecas necessárias para o código. `pandas` e `numpy` são usadas para manipulação de dados, enquanto `sklearn` é usada para modelagem de aprendizado de máquina:

```python
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

* Os dados são carregados do arquivo "iris.data" para um DataFrame do pandas chamado `dados`:

```python
dados = pd.read_csv("iris.data")
```

* Esta linha remove qualquer linha do DataFrame que tenha pelo menos um valor faltante:

```python
dados.dropna(inplace=True)
```

* O DataFrame é dividido em duas partes: `X` e `y`. `X` contém os atributos (as primeiras quatro colunas do DataFrame), e `y` contém as classes (a coluna 'class'):

```python
X = np.array(dados.iloc[:, 0:3])
y = np.array(dados['class'])
```

* Esta linha divide os dados em conjuntos de treinamento e teste. 70% dos dados são usados para treinamento (`X_train` e `Y_train`), e 30% dos dados são usados para teste (`X_test` e `Y_test`):

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, train_size=0.7)
```

* Um modelo de classificador K-Nearest Neighbors (KNN) é criado com `neighboor` vizinhos. Em seguida, o modelo é ajustado aos dados de treinamento:

```python
neighboor = 7
knn = KNeighborsClassifier(neighboor)
knn.fit(X_train, Y_train)
```

* Esta linha usa o modelo treinado para fazer previsões nos dados de teste:

```python
previsoes = knn.predict(X_test)
```

* A acurácia do modelo é calculada comparando as previsões do modelo com as classes reais dos dados de teste:

```python
acuracia = accuracy_score(Y_test, previsoes) * 100
```

* Imprime a acurácia do modelo:

```python
print("A acurácia foi %.2f%%" % acuracia)
```
