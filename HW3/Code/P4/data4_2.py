import pandas as pd
import numpy as np

def pca(X, n_components=50):
    mean_vec = np.mean(X, axis=0)
    centered_X = X - mean_vec

    U, S, V = np.linalg.svd(centered_X, full_matrices=False)

    # Sorting
    idx = np.argsort(S)[::-1]
    U = U[:, idx]
    S = S[idx]
    V = V[idx, :]

    # Reduce dimensionality
    U = U[:, :n_components]
    S = np.diag(S[:n_components])
    V = V[:n_components, :]
    
    return U, S, V, mean_vec

# Load data
filepath = '/home/sprutz/dev/mlp/HW3/HW3_data/P4_files/spam_ham.csv'
data = pd.read_csv(filepath)
data.drop('Unnamed: 0', axis=1, inplace=True)

X = data.drop('cls', axis=1).to_numpy()
y = data['cls'].to_numpy()

# Perform PCA
U, S, V, mean_vec = pca(X)

X_reduced = np.dot(U, S)

# Splitting data
idx = np.arange(X_reduced.shape[0])
np.random.shuffle(idx)

X_reduced = X_reduced[idx]
y = y[idx]

X_train, X_test = X_reduced[:3500], X_reduced[3500:]
y_train, y_test = y[:3500], y[3500:]

def main():
    # Saving data
    np.savez('/home/sprutz/dev/mlp/HW3/HW3_data/P4_files/train4_2.npz', x=X_train, y=y_train)
    np.savez('/home/sprutz/dev/mlp/HW3/HW3_data/P4_files/test4_2.npz', x=X_test, y=y_test)

if __name__ == "__main__":
    main()