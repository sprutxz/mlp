import numpy as np
import matplotlib.pyplot as plt

X = np.load('../../HW2_data/P3_data/train_100.npz')['x']
y = np.load('../../HW2_data/P3_data/train_100.npz')['y']

def prep_data(X, degree = 9):
    X = np.vstack([X**i for i in range(degree+1)]).T
    return X

X = prep_data(X)

w = np.linalg.inv(X.T @ X) @ X.T @ y

plot_x = np.linspace(0, 1, 100)

plot_X = prep_data(plot_x)

plot_y = np.dot(plot_X, w)

plt.scatter(X[:, 1], y, label='data points')
plt.plot(plot_x, plot_y, color='red', label='fitted curve')
plt.legend()
plt.show()