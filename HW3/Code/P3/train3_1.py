import numpy as np

filepath = '/home/sprutz/dev/mlp/HW3/HW3_data/P3_data/data_1/train.npz'
train = np.load(filepath)
X = train['x']
y = train['y']

mean_pos = np.mean(X[y == 1], axis=0)
mean_neg = np.mean(X[y == -1], axis=0)

var_pos = np.var(X[y == 1], axis=0)
var_neg = np.var(X[y == -1], axis=0)

# calculating frequency of classes for use in test3_3.py
freq_pos = np.mean(y == 1)
freq_neg = np.mean(y == -1)

def main():
    print(mean_pos, var_pos)
    print(mean_neg, var_neg)

if __name__ == "__main__":
    main()