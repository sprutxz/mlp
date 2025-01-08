import pandas as pd
import numpy as np

dataset = pd.read_csv('DiabetesData/train.csv')
dataset.drop('Unnamed: 0', axis=1, inplace=True)

X = dataset[['glucose', 'bloodpressure']].values
y = dataset['diabetes'].values
X_negative = X[y == 0]
X_positive = X[y == 1]

def gaussian_density(x, mean, var):
    coef = 1.0 / np.sqrt(2.0 * np.pi * var)
    exponent = np.exp(- (x - mean) ** 2 / (2 * var))
    return coef * exponent

mean_pos = np.mean(X_positive, axis=0)
var_pos = np.var(X_positive, axis=0)

mean_neg = np.mean(X_negative, axis=0)
var_neg = np.var(X_negative, axis=0)

glucose_densities_pos = gaussian_density(X[:, 0], mean_pos[0], var_pos[0])
bp_densities_pos = gaussian_density(X[:, 1], mean_pos[1], var_pos[1])

glucose_densities_neg = gaussian_density(X[:, 0], mean_neg[0], var_neg[0])
bp_densities_neg = gaussian_density(X[:, 1], mean_neg[1], var_neg[1])