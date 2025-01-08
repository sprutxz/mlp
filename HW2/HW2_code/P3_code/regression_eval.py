import numpy as np
import matplotlib.pyplot as plt

degree = 9

def prep_data(X, degree):
    X = np.vstack([X**i for i in range(degree+1)]).T
    return X

# loading models
ols_models = []
for i in range(5):
    model = np.load(f"../../models/w(L=0){i}.npy")
    ols_models.append(model)

ridge_models = []
for i in range(5):
    model = np.load(f"../../models/w(la=0.0001){i}.npy")
    ridge_models.append(model)

# averaging models
ols_avg = np.mean(ols_models, axis=0)
ridge_avg = np.mean(ridge_models, axis=0)

print(ols_avg.shape)
print(ridge_avg.shape)

#preparing plot data
X = np.linspace(0, 1, 100)
X = prep_data(X, degree)

#plotting
plt.figure()
plt.plot(X[:,1], np.dot(X, ols_avg), label="OLS")
plt.plot(X[:,1], np.dot(X, ridge_avg), label="Ridge")
plt.legend()
plt.show()

# evaluting
test_X = np.load("../../HW2_data/P3_data/test.npz")["x"]
test_y = np.load("../../HW2_data/P3_data/test.npz")["y"]

# preparing test data
test_X = prep_data(test_X, degree)

ols_test_error = np.mean((test_y - np.dot(test_X, ols_avg))**2)
ridge_test_error = np.mean((test_y - np.dot(test_X, ridge_avg))**2)

print(f"OLS test error: {ols_test_error}")
print(f"Ridge test error: {ridge_test_error}")