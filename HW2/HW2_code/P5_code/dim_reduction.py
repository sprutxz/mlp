import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

X = np.load("/home/sprutz/dev/mlp/HW2/HW2_data/P5_data/vgg16_train.npz")["logit"]
y = np.load("/home/sprutz/dev/mlp/HW2/HW2_data/P5_data/vgg16_train.npz")["year"]

meanv = np.mean(X, axis = 0)
X_cntr = X - meanv

cov_matrix = np.cov(X_cntr, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

whitening_matrix = np.dot(eigenvectors, np.diag(1.0 / np.sqrt(eigenvalues)))
X_whitened = np.dot(X_cntr, whitening_matrix)
X_whitened_1D = X_whitened[:, 0]
X_whitened_2D = X_whitened[:, :2]

# fig = plt.figure()
# # ax = fig.add_subplot(projection = '3d')
# ax = fig.add_subplot(111)
# cmap = mpl.cm.viridis
# norm = mpl.colors.Normalize(vmin=1148, vmax=2012)
# fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=plt.gca())
# # ax.scatter(X_whitened_2D[:, 0], X_whitened_2D[:, 1], y, c=y, s=2, picker=4)
# ax.scatter(X_whitened_1D, y, c=y, s=2, picker=4)
# plt.show()