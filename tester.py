from copy import deepcopy
import utils
import numpy as np

class Tester:

    def __init__(self, dataset, model):
        self.model = model
        self.dataset = dataset
        self.conf_matrices = []
        
    def test(self, n_tests=100, transform=None):
        for _ in range(n_tests):
            X_train, Y_train, X_test, Y_test = self.dataset.get_random_train_test()
            
            T = None
            if transform == 'PCA':
                T = utils.PCA(X_train, 0.999)
            elif transform == 'LDA':
                T = utils.LDA(X_train, Y_train, 0.999)
            if T:
                X_train = T.transform(X_train)
                X_test = T.transform(X_test)
                
            model = deepcopy(self.model)
            model.train(X_train, Y_train)
            
            cm = np.zeros((2, 2))
            
            for x, y in zip(X_test, Y_test):
                pred = model.predict(x)
                cm[y, pred] += 1
            
            self.conf_matrices.append(cm)
     
    def statistics(self):
        return Statistics(self)
        
class Statistics:
    def __init__(self, tester):
        if not tester.conf_matrices:
            print('Not tested')
        else:
            self.m = np.array(tester.conf_matrices)
            self.m_sum = self.m.sum(axis=0)
    
    def mean_accuracy(self):
        return self.m_sum.trace() / self.m_sum.sum()
    
    def specificity(self):
        return self.m_sum[0, 0] / self.m_sum.sum(axis=0)[0]
    
    def sensibility(self):
        return self.m_sum[1, 1] / self.m_sum.sum(axis=0)[1]
    
    def print_all(self):
        print(self.m_sum)
        print('Mean accuracy: {}'.format(self.mean_accuracy()))
        print('Specificity: {}'.format(self.specificity()))
        print('Sensibility: {}'.format(self.sensibility()))
