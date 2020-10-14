import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))  # any rows , one column
print("X:",x)
y = np.array([5, 20, 14, 32, 22, 38])
print("Y:",y)

model = LinearRegression()
model.fit(x, y)
score = model.score(x, y)
prediction = model.predict(x)

print('coefficient of determination:', score)
print("y intercept", model.intercept_)
print("slope", model.coef_)
print('predicted ', prediction)

x_new = np.arange(5).reshape((-1, 1))#generate elements from 0 up to but not including 5
y_new = model.predict(x_new)
print(y_new)
