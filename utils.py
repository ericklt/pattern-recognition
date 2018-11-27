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

def pooled_cov(gaussians):
    return np.average([g.cov for g in gaussians], axis=0, weights=[g.n for g in gaussians])

def __friedman_for_gaussian(g, total, pooled, alpha):
    return np.average(np.array([g.cov, pooled]), axis=0, weights=[(1 - alpha) * g.n, alpha * total])
        
def apply_friedman(gaussians, alpha):
    total = sum([g.n for g in gaussians])
    pooled = pooled_cov(gaussians)
    for g in gaussians:
        g.update_cov(__friedman_for_gaussian(g, total, pooled, alpha))

class Gaussian:
    def __init__(self, X, naive=False):
        self.X = X
        self.n = len(X)
        self.mean = np.mean(X, axis=0)
        c = cov(X)
        if naive:
            c = np.multiply(c, np.eye(c.shape[0]))
        self.update_cov(c)
        
    def singular(self):
        return np.linalg.matrix_rank(self.cov) != self.cov.shape[0]
        
    def update_cov(self, cov):
        self.cov = cov
        self.cov_inv = np.linalg.pinv(cov)
        self.cov_det = np.linalg.det(cov)
        
    def log_prob(self, x):
        z = x - self.mean
        return - 0.5 * (np.dot(z, np.dot(self.cov_inv, z)) + np.log(self.cov_det))

class Transformer:
    
    def __init__(self):
        self.eig_vals = None
        self.T = None
    
    def transform(self, X, min_var):
        if self.T is None:
            print('Not Fit')
            return
        cs = np.cumsum(self.eig_vals / self.eig_vals.sum())
        n_components = len([x for x in cs if x < min_var]) + 1
        subT = self.T[:, :n_components]
        return np.matmul(X, subT)

class PCA(Transformer):

    def fit(self, X, Y=None):
        vals, vecs = np.linalg.eig(cov(X))
        idx = vals.argsort()[::-1]
        self.eig_vals, self.T = vals[idx], np.real(vecs[:, idx])
        
class LDA(Transformer):
    
    def fit(self, X, Y):
        
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

    