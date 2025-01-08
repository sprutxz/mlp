import numpy as np
import matplotlib.pyplot as plt

X = np.load("../../HW2_data/P3_data/train.npz")["x"]
y = np.load("../../HW2_data/P3_data/train.npz")["y"]

# adding basis function
degree = 9
k = 5

X = np.vstack([X**i for i in range(degree+1)]).T

print(f"train_x shape: {X.shape} \ntrain_y shape: {y.shape}")

# shuffle data
# idx = np.random.permutation(X.shape[0])
# X = X[idx]
# y = y[idx]

# split data into k folds
fold_x = np.array_split(X, k)
fold_y = np.array_split(y, k)

lambdas = [1, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001]

# 5 fold cv
loss = []
models = []

for la in lambdas:
    val_error = []
    model_group = []
    for i in range(k):
        train_x = np.vstack([fold_x[j] for j in range(k) if j != i])
        train_y = np.concatenate([fold_y[j] for j in range(k) if j != i])
        test_x = fold_x[i]
        test_y = fold_y[i]
        
        w = np.linalg.inv(train_x.T @ train_x + la*np.eye(degree+1)) @ train_x.T @ train_y
        model_group.append(w)
            
        y_pred = np.dot(test_x, w)
        err = np.mean((test_y - y_pred)**2)
        val_error.append(err)
    
    models.append(model_group)
        
    loss.append(np.mean(val_error))
        
print("Average validation error:", np.mean(loss))

print("Best lambda:", lambdas[np.argmin(loss)])

for i, model in enumerate(models[np.argmin(loss)]):
    np.save(f"../../models/w(la={lambdas[np.argmin(loss)]}){i}.npy", model)

# plotting loss
plt.figure()
plt.scatter(lambdas, loss, label='Validation Error')
plt.xlabel("lambda")
plt.ylabel("Validation error")
plt.title("Validation Error vs Lambda")
plt.legend()
plt.show()


