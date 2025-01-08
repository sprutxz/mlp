import numpy as np

filepath = '/home/sprutz/dev/mlp/HW3/HW3_data/P3_data/data_2/train.npz'
train = np.load(filepath)\
    
X = train['x']
y = train['y']

mean_pos = np.mean(X[y == 1], axis=0)
mean_neg = np.mean(X[y == -1], axis=0)

cov_pos = np.cov(X[y == 1], rowvar=False)
cov_neg = np.cov(X[y == -1], rowvar=False)

def main():
    print(mean_pos, mean_neg)
    print(cov_pos)
    print(cov_neg)

    print(np.mean(y == 1), np.mean(y == -1))

if __name__ == "__main__":
    main()
