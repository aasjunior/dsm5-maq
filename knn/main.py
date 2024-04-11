from models.DataModel import DataModel
from models.KNNModel import KNNModel

numeric_cols = ['sepal_length','sepal_width','petal_length','petal_width']
categorical_cols = ['class']

model = DataModel("db/iris.data", numeric_cols, categorical_cols)

# print(model.data)
model.normalize_data()
# print(model.data)

knn = KNNModel(model, 7)
knn.train_and_evaluate()