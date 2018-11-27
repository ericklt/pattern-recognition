import numpy as np
import utils

def parkinsons(normalized=False):
    data = np.loadtxt('data/parkinsons.csv', delimiter=',', skiprows=1, usecols=range(1, 24))
    X = np.delete(data, 16, axis=1)
    Y = data[:, 16].astype(int)
    return Dataset(X, Y, normalized)

def credit_cards(normalized=False):
    data = np.loadtxt('data/credit_card_clients.csv', delimiter=',', skiprows=2)
    X = data[:, 1:-1]
    Y = data[:, -1].astype(int)
    return Dataset(X, Y, normalized)

class Dataset:
    def __init__(self, X, Y, normalized=False):
        self.X = X
        self.Y = Y
        if normalized:
            self.X = (self.X - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)
        
        self.cd = utils.to_class_dict(self.X, self.Y)
        
    def get_random_train_test(self, train_percentage=0.8):
        splt = int(train_percentage * self.X.shape[0])
        perm = np.random.permutation(range(self.X.shape[0]))
        X_, Y_ = self.X[perm], self.Y[perm]
        return X_[:splt], Y_[:splt], X_[splt:], Y_[splt:]
        
    def get_stratified_train_test(self, train_percentage=0.8):
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
    