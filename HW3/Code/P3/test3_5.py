import numpy as np
from train3_4 import *

filepath = '/home/sprutz/dev/mlp/HW3/HW3_data/P3_data/data_2/test.npz'
test = np.load(filepath)

X = test['x']
y = test['y']

def formula(x, mean, cov):
    # I first compute the exponent
    diff = x - mean
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)
    
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    return np.exp(exponent) / np.sqrt((2 * np.pi) ** 2 * cov_det)


pred = np.where(formula(X, mean_pos, cov_pos) > formula(X, mean_neg, cov_neg), 1, -1)
print(np.mean(pred == y))