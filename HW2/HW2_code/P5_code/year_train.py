import numpy as np
from dim_reduction import *
from itertools import combinations_with_replacement

X = X_whitened_2D
y = y

# splitting data into 80 20
split = int(X.shape[0] * 0.2)
indices = np.random.permutation(X.shape[0])
val_indices = indices[:split]
train_indices = indices[split:]

X_train = X[train_indices]
y_train = y[train_indices]
X_val = X[val_indices]
y_val = y[val_indices]

print(X_train.shape, y_train.shape)

max_degree = 5

# adding polynomial features
def polynomial_features(X, degree):
    n_samples, n_features = X.shape
    features = [np.ones(n_samples)]
    for d in range(1, degree + 1):
        for comb in combinations_with_replacement(range(n_features), d):
            features.append(np.prod(X[:, comb], axis=1))
    return np.column_stack(features)

mse_list = []
best_mse = 0
best_weights = None
best_degree = 0

for degree in range(1, max_degree + 1):
    X_train_poly = polynomial_features(X_train, degree)
    X_val_poly = polynomial_features(X_val, degree)
    print(X_train_poly.shape, X_val_poly.shape)
    
    w = np.linalg.inv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train
    y_pred = X_val_poly @ w
    mse = np.mean((y_val - y_pred) ** 2)
    mse_list.append(mse)
    
    if mse < best_mse or best_mse == 0:
        best_mse = mse
        best_degree = degree
        best_weights = w

print("Best degree:", best_degree)
print("Best val MSE:", best_mse)