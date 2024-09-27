import numpy as np
from sklearn.linear_model import LinearRegression


#the input, x, and the output, y. You should call .reshape() on x because this array must be two-dimensional,
# or more precisely, it must have one column and as many rows as necessary. That’s exactly what the argument (-1, 1)
#of .reshape() specifies.

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])


model = LinearRegression()
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")

print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")

y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")
