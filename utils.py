import numpy as np

def cov(X):
    z = X - np.mean(X, axis=0)
    return (z.T.dot(z)) / (z.shape[0] - 1)

def distance(x1, x2):
    return np.linalg.norm(x2 - x1)

def to_class_dict(X, Y):
    classes_dict = {}
    for x, y in zip(X, Y):
        if y not in classes_dict:
            classes_dict[y] = []
        classes_dict[y].append(x)
    return {y : np.array(classes_dict[y]) for y in classes_dict} 

class Transformer:
    
    def __init__(self):
        self.min_var = None
        self.eig_vals = None
        self.T = None
    
    def transform(self, X):
        cs = np.cumsum(self.eig_vals / self.eig_vals.sum())
        n_components = len([x for x in cs if x < self.min_var]) + 1
        subT = self.T[:, :n_components]
        return np.matmul(X, subT)

class PCA(Transformer):
    
    def __init__(self, X, min_var):
        self.min_var = min_var
        
        vals, vecs = np.linalg.eig(cov(X))
        idx = vals.argsort()[::-1]
        self.eig_vals, self.T = vals[idx], np.real(vecs[:, idx])
        
class LDA(Transformer):
    
    def __init__(self, X, Y, min_var):
        self.min_var = min_var
        
        cd = to_class_dict(X, Y)
        classes = [cd[y] for y in sorted(cd)]

        m = np.mean(np.array([x for c in classes for x in c]), axis=0)
        centroids = [np.mean(c, axis=0) for c in classes]
        covariances = [cov(c) for c in classes]

        Sw = sum([len(classes[i]) * covariances[i] for i in range(len(classes))])
        Sw_inv = np.linalg.inv(Sw)
        n_m = [(c - m).reshape(1, -1) for c in centroids]
        Sb = sum([np.dot(n_m[i].T, n_m[i]) for i in range(len(classes))])

        vals, vecs = np.linalg.eig(np.dot(Sw_inv, Sb))
        idx = vals.argsort()[::-1]
        self.eig_vals, self.T = vals[idx], np.real(vecs[:, idx])

    