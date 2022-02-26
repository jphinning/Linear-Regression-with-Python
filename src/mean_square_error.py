""" Linear Regression"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
X_REAL_DATA = [0, 0.1, 0.2, 0.3, 0.4]
Y_REAL_DATA = [0.2, 0.3, 0.45, 0.7, 0.8]

x = np.array(X_REAL_DATA).reshape((-1, 1))
y = np.array(Y_REAL_DATA)

# Calculate coeficient a and intersection b
linear_model = LinearRegression().fit(x,y)

# Store a and b in these variables
coef_a = linear_model.coef_
intersection_b = linear_model.intercept_

# Generate y = ax + b values
predicted_data_x = np.linspace(0, 0.4)
predicted_data_y = coef_a * predicted_data_x + intersection_b

# Ploting scattered and simple charts
fig, ax = plt.subplots()

ax.scatter(X_REAL_DATA, Y_REAL_DATA)
ax.plot(predicted_data_x, predicted_data_y)

ax.set(xlabel='xi', ylabel='yi', title='Linear Regression')
ax.grid()
fig.savefig("linearRegression.png")
