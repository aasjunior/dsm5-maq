from models.DataModel import DataModel
from models.KNNModel import KNNModel

numeric_cols = ['sepal_length','sepal_width','petal_length','petal_width']
categorical_cols = ['class']

model = DataModel("db/iris.data", numeric_cols, categorical_cols)

# print(model.data.head())
data = model.normalize_data()

knn = KNNModel(model.data, 7)
knn.train_and_evaluate()