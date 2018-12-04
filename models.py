from scipy import spatial
import numpy as np
import utils
from sklearn.neural_network import MLPClassifier

class MLP:
    
    def __init__(self, layers=(10, 3), max_iter=200):
        self.X = None
        self.Y = None
        self.mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=max_iter)
        
    def fit(self, X, Y):
        self.mlp.fit(X, Y)
        
    def predict(self, X):
        return self.mlp.predict(X)

class KNN:
    
    def __init__(self, k=1):
        self.k = k
        self.X = None
        self.Y = None
        self.kdTree = None
    
    def fit(self, X_train, Y_train):
        self.X = X_train
        self.Y = Y_train
        self.kdTree = spatial.cKDTree(self.X, leafsize=10)
        
    def predict(self, X):
        if self.X is None:
            raise Exception('Not trained')
        return [self.__predict_one(x) for x in X]
     
    def __predict_one(self, x):
        query = self.kdTree.query(x, k=self.k)
        if self.k == 1:
            return self.Y[query[1]]
        classes = np.array([self.Y[i] for i in query[1]])
        return np.bincount(classes).argmax()


class DMC:
    
    def __init__(self):
        self.clusters = {}
        self.centroids = None
    
    def fit(self, X_train, Y_train):
        self.clusters = utils.to_class_dict(X_train, Y_train)
        self.centroids = {k : np.mean(self.clusters[k], axis=0) for k in self.clusters}
        
    def predict(self, X):
        if self.centroids is None:
            raise Exception('Not trained')
        return [self.__predict_one(x) for x in X]
        
    def __predict_one(self, x):
        return min(self.centroids, key=lambda c : utils.distance(self.centroids[c], x))
            
class CQG:
    
    def __init__(self, naive=False, friedman_value=1):
        self.gaussians = None
        self.a_priori = None
        self.naive = naive
        self.friedman_value = friedman_value

    def a_posteriori(self, x):
        return np.array([g.log_prob(x) for g in self.gaussians])
    
    def fit(self, X_train, Y_train):
        cd = utils.to_class_dict(X_train, Y_train)
        self.gaussians = [utils.Gaussian(cd[y], self.naive) for y in sorted(cd.keys())]
        
        if any([g.singular() for g in self.gaussians]):
            utils.apply_friedman(self.gaussians, self.friedman_value)
            
        self.a_priori = np.array([ np.log(g.n) for g in self.gaussians ])
        
    def predict(self, X):
        if self.gaussians is None:
            raise Exception('Not trained')
        return [self.__predict_one(x) for x in X]
        
    def __predict_one(self, x):
        probabilities = self.a_priori + self.a_posteriori(x)
        return probabilities.argmax()