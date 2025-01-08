import numpy as np
from year_train import *

test_logit=np.load("/home/sprutz/dev/mlp/HW2/HW2_data/P5_data/vgg16_test.npz")["logit"]
test_year=np.load("/home/sprutz/dev/mlp/HW2/HW2_data/P5_data/vgg16_test.npz")["year"]
test_filename=np.load("/home/sprutz/dev/mlp/HW2/HW2_data/P5_data/vgg16_test.npz", allow_pickle=True)["filename"]

x_test_cntr = test_logit - meanv
whitening_matrix = np.dot(eigenvectors, np.diag(1.0 / np.sqrt(eigenvalues)))
x_test_whitened = np.dot(x_test_cntr, whitening_matrix)
x_test_whitened_2D = x_test_whitened[:, :2]

X_test_poly = polynomial_features(x_test_whitened_2D, best_degree)
y_pred = X_test_poly @ best_weights
mse = np.mean((test_year - y_pred) ** 2)

print("Test MSE:", mse)

errors = np.abs(test_year - y_pred)
print(f"Most accurate prediction: {test_filename[np.argmin(errors)]} as {test_year[np.argmin(errors)]}, predicted {y_pred[np.argmin(errors)]}")
print(f"Least accurate prediction: {test_filename[np.argmax(errors)]} as {test_year[np.argmax(errors)]}, predicted {y_pred[np.argmax(errors)]}")