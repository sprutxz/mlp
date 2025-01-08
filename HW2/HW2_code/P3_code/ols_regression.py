import numpy as np

X = np.load("../../HW2_data/P3_data/train.npz")["x"]# 25 data points
y = np.load("../../HW2_data/P3_data/train.npz")["y"]

print(f"train_x shape: {X.shape} \ntrain_y shape: {y.shape}")

# adding basis function
degree = 9
k = 5

X = np.vstack([X**i for i in range(degree+1)]).T

# shuffle data
# idx = np.random.permutation(X.shape[0])
# X = X[idx]
# y = y[idx]

# split data into k folds
fold_x = np.array_split(X, k)
fold_y = np.array_split(y, k)

# 5 fold cv
validation_error = []
for i in range(k):
    train_x = np.vstack([fold_x[j] for j in range(k) if j != i])
    train_y = np.concatenate([fold_y[j] for j in range(k) if j != i])
    test_x = fold_x[i]
    test_y = fold_y[i]
    
    w = np.linalg.inv(train_x.T @ train_x) @ train_x.T @ train_y
    np.save(f"../../models/w(L=0){i}.npy", w)
    
    print(f"Fold {i+1}:")
    print(f"Training error: {np.mean((train_y - np.dot(train_x, w))**2)}")
    print(f"Validation error: {np.mean((test_y - np.dot(test_x, w))**2)}")
    print()
        
    y_pred = np.dot(test_x, w)
    validation_error.append(np.mean((test_y - y_pred)**2))
    
print("Average validation error:", np.mean(validation_error))