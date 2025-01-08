import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(X, y, w):
    z = np.dot(X, w)
    predictions = sigmoid(z)
    nll = -np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return nll

def compute_gradient(X, y, w):
    z = np.dot(X, w)
    predictions = sigmoid(z)
    gradient = np.dot(X.T, (predictions - y))
    return gradient

def train_logistic_regression(X, y, learning_rate=1e-1, tolerance=1e-10, max_iter=100000):
    w = np.zeros(X.shape[1])
    
    prev_nll = float('inf')
    
    for i in range(max_iter):
        nll = loss(X, y, w)
        
        if abs(prev_nll - nll) < tolerance:
            print(f"Converged after {i} iterations with NLL = {nll}")
            break
        
        gradient = compute_gradient(X, y, w)
        
        w -= learning_rate * gradient
        
        prev_nll = nll
        
        if i % 1000 == 0:
            print(f"Iteration {i}: NLL = {nll}")
    
    return w

def main():
    train_file = '/home/sprutz/dev/mlp/HW3/HW3_data/P4_files/train4_2.npz'
    train = np.load(train_file)
    X_train = train['x']
    y_train = train['y']

    test_file = '/home/sprutz/dev/mlp/HW3/HW3_data/P4_files/test4_2.npz'
    test = np.load(test_file)
    X_test = test['x']
    y_test = test['y']

    w = train_logistic_regression(X_train, y_train)

    # accuracy for train
    z = np.dot(X_train, w)
    predictions = sigmoid(z)
    predictions = np.round(predictions)
    accuracy = np.mean(predictions == y_train)
    print(f"Accuracy: {accuracy}")

    # accuracy for test
    z = np.dot(X_test, w)
    predictions = sigmoid(z)
    predictions = np.round(predictions)
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy}")

    # saving the weights
    np.save('/home/sprutz/dev/mlp/HW3/models/q4weights.npy', w)

if __name__ == "__main__":
    main()