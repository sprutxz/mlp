import numpy as np
from train3_4 import *

filepath = '/home/sprutz/dev/mlp/HW3/HW3_data/P3_data/data_2/test.npz'
test = np.load(filepath)
X = test['x']
y = test['y']

# Function to compute the Gaussian likelihood
def formula(x, mean, cov):
    diff = x - mean
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)
    
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    return np.exp(exponent) / np.sqrt((2 * np.pi) ** 2 * cov_det)

mean_pos = np.array([0, 0])
mean_neg_1 = np.array([0, 2])
mean_neg_2 = np.array([0, -2])
cov_pos = np.eye(2)
cov_neg = np.eye(2)

likelihood_pos = formula(X, mean_pos, cov_pos)

likelihood_neg_1 = formula(X, mean_neg_1, cov_neg)
likelihood_neg_2 = formula(X, mean_neg_2, cov_neg)

likelihood_neg = 0.5 * likelihood_neg_1 + 0.5 * likelihood_neg_2

pred = np.where(likelihood_pos >= likelihood_neg, 1, -1)

# Compute and print the accuracy
accuracy = np.mean(pred == y)
print(f"Accuracy: {accuracy}")