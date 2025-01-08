"""
The entiry of gaussian density calcualtion is done again in this file because
python doesn't allow importing of code from files with '-' in the name.
"""

import pandas as pd
import numpy as np

def gaussian_density(x, mean, var):
    coef = 1.0 / np.sqrt(2.0 * np.pi * var)
    exponent = np.exp(- (x - mean) ** 2 / (2 * var))
    return coef * exponent

dataset = pd.read_csv('/home/sprutz/dev/mlp/DiabetesData/train.csv')
dataset.drop('Unnamed: 0', axis=1, inplace=True)

X = dataset[['glucose', 'bloodpressure']].values
y = dataset['diabetes'].values

# standardize the data
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std
# mean = np.mean(X, axis=0)
# std = np.std(X, axis=0)
# print(mean, std)

X_negative = X[y == 0]
X_positive = X[y == 1]

mean_pos = np.mean(X_positive, axis=0)
var_pos = np.var(X_positive, axis=0)

mean_neg = np.mean(X_negative, axis=0)
var_neg = np.var(X_negative, axis=0)
# print(mean_pos, var_pos)
# print(mean_neg, var_neg)


# Calculate Gaussian densities for positive class (D = +)
glucose_densities_pos = gaussian_density(X[:, 0], mean_pos[0], var_pos[0])
bp_densities_pos = gaussian_density(X[:, 1], mean_pos[1], var_pos[1])

# Calculate Gaussian densities for negative class (D = -)
glucose_densities_neg = gaussian_density(X[:, 0], mean_neg[0], var_neg[0])
bp_densities_neg = gaussian_density(X[:, 1], mean_neg[1], var_neg[1])

# Calculate the likelihoods and prior probabilities for postive class
likelihoods_pos = glucose_densities_pos * bp_densities_pos
prior_pos = len(X_positive) / len(X)

# Calculate the likelihoods and prior probabilities for negative class
likelihoods_neg = glucose_densities_neg * bp_densities_neg
prior_neg = len(X_negative) / len(X)

# Predict the class labels
predictions = likelihoods_pos * prior_pos > likelihoods_neg * prior_neg
predictions = predictions.astype(int)

calculate_accuracy = lambda y_true, y_pred: np.mean(y_true == y_pred)
accuracy = calculate_accuracy(y, predictions)
print(f"Accuracy train: {accuracy:.3f}")

# Evaluate the model on the test set
test_dataset = pd.read_csv('/home/sprutz/dev/mlp/DiabetesData/test.csv')
test_dataset.drop('Unnamed: 0', axis=1, inplace=True)

X_test = test_dataset[['glucose', 'bloodpressure']].values
y_test = test_dataset['diabetes'].values

# standardize the data
X_test = (X_test - mean) / std

# Calculate Gaussian densities for positive class (D = +)
glucose_densities_pos_test = gaussian_density(X_test[:, 0], mean_pos[0], var_pos[0])
bp_densities_pos_test = gaussian_density(X_test[:, 1], mean_pos[1], var_pos[1])

# Calculate Gaussian densities for negative class (D = -)
glucose_densities_neg_test = gaussian_density(X_test[:, 0], mean_neg[0], var_neg[0])
bp_densities_neg_test = gaussian_density(X_test[:, 1], mean_neg[1], var_neg[1])

# Calculate the likelihoods
likelihoods_pos_test = glucose_densities_pos_test * bp_densities_pos_test
likelihoods_neg_test = glucose_densities_neg_test * bp_densities_neg_test

# Calculate the posterior probabilities
posterior_pos_test = likelihoods_pos_test * prior_pos
posterior_neg_test = likelihoods_neg_test * prior_neg

# Predict the class labels
predictions_test = posterior_pos_test > posterior_neg_test
predictions_test = predictions_test.astype(int)

accuracy_test = calculate_accuracy(y_test, predictions_test)
print(f"Accuracy test: {accuracy_test:.3f}")