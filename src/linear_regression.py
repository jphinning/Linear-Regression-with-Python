""" Linear Regression"""
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 0.1, 0.2, 0.3, 0.4])
y = np.array([0.2, 0.3, 0.45, 0.7, 0.8])

A = np.vstack([x, np.ones(len(x))]).T

m, c = np.linalg.lstsq(A, y, rcond=None)[0]

# Ploting scattered and simple charts
fig, ax = plt.subplots()

plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, m*x + c, 'r', label='Fitted line')
plt.legend()
ax.set(xlabel='xi', ylabel='yi', title='Linear Regression')
ax.grid()

fig.savefig("linearRegression.png")
