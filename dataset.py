import numpy as np
import utils

class Parkinsons:
    def __init__(self, normalized=False):
        self.data = np.loadtxt('parkinsons.csv', delimiter=',', skiprows=1, usecols=range(1, 24))
        X = np.delete(self.data, 16, axis=1)
        self.X = X - np.mean(X, axis=0)
        if normalized:
            self.X /= np.linalg.norm(self.X, axis=0)
        self.Y = self.data[:, 16].astype(int)
        self.cd = utils.to_class_dict(self.X, self.Y)
        
    def get_random_train_test(self, train_percentage=0.8):
        shuffled_cd = {y : np.random.permutation(self.cd[y]) for y in self.cd}
        class_splits = {y: int(train_percentage * len(self.cd[y])) for y in shuffled_cd}
        train_cd = {y: shuffled_cd[y][:class_splits[y]] for y in shuffled_cd}
        test_cd = {y: shuffled_cd[y][class_splits[y]:] for y in shuffled_cd}
        train = np.random.permutation([[x, y] for y in train_cd for x in train_cd[y]])
        test = np.random.permutation([[x, y] for y in test_cd for x in test_cd[y]])
        X_train = np.array([k[0] for k in train])
        Y_train = np.array([k[1] for k in train])
        X_test = np.array([k[0] for k in test])
        Y_test = np.array([k[1] for k in test])
        return X_train, Y_train, X_test, Y_test
    