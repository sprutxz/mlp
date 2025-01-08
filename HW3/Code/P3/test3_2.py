import numpy as np
from train3_1 import *

filepath = '/home/sprutz/dev/mlp/HW3/HW3_data/P3_data/data_1/test.npz'
test = np.load(filepath)
X = test['x']
y = test['y']

def formula(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(- (x - mean) ** 2 / (2 * var))

pred = np.where(formula(X, mean_pos, var_pos) > formula(X, mean_neg, var_neg), 1, -1)
print(np.mean(pred == y))
