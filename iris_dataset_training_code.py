from numpy.core.fromnumeric import size
from numpy.lib.function_base import gradient
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

data = load_iris()
df = pd.DataFrame(data["data"], columns=data["feature_names"])
df["species"] = data["target"]
print("your data is :")
print(df)
print("\n")

from sklearn.model_selection import train_test_split

x = df.drop('species', axis=1)
y = df.species
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)
predictions = np.round(model.predict(x_test))
print("predictions using sklearn linear regression are :")
print(predictions)
print("\n")

from sklearn.metrics import accuracy_score
from numpy import array
print("accuracy score for sklearn linear regression is :")
print(accuracy_score(array(y_test), predictions))
print("\n")

from numpy.linalg import inv

X_m_train = array(x_train)
Y_m_train = array(y_train)
X_m_test = array(x_test)
Y_m_test = array(y_test)

W = inv(X_m_train.T.dot(X_m_train)).dot(X_m_train.T).dot(Y_m_train)

new_predictions = np.round(X_m_test.dot(W))
print("predictions using numpy linear regression are :")
print(new_predictions)
print("\n")

print("accuracy score for numpy linear regression is :")
print(accuracy_score(Y_m_test, new_predictions))
print('\n')
