from copy import deepcopy
import utils
import numpy as np
import pandas as pd
from IPython.display import display, HTML

class Tester:

    def __init__(self, dataset, model):
        self.model = model
        self.dataset = dataset
        self.conf_matrices = []
        
    def test(self, n_tests=100, transform=None, min_transform_var=0.999):
        for _ in range(n_tests):
            X_train, Y_train, X_test, Y_test = self.dataset.get_random_train_test()
            
            if transform:
                transform.fit(X_train, Y_train)
                X_train = transform.transform(X_train, min_transform_var)
                X_test = transform.transform(X_test, min_transform_var)
                
            model = deepcopy(self.model)
            model.fit(X_train, Y_train)
            
            cm = np.zeros((2, 2))
            
            preds = model.predict(X_test)
            
            for y, pred in zip(Y_test, preds):
                cm[y, pred] += 1
            
            self.conf_matrices.append(cm)
     
    def statistics(self):
        return Statistics(self)
        
class Statistics:
    def __init__(self, tester):
        if not tester.conf_matrices:
            print('Not tested')
        else:
            self.ms = np.array(tester.conf_matrices)
            self.m_sum = self.ms.sum(axis=0)
            self.accs = np.array([m.trace() / m.sum() for m in self.ms])
    
    def acc_mean(self):
        return np.mean(self.accs)
    
    def acc_median(self):
        return np.median(self.accs)
    
    def acc_max(self):
        return np.max(self.accs)
    
    def acc_min(self):
        return np.min(self.accs)
    
    def acc_std(self):
        return np.std(self.accs)
    
    def specificity(self):
        return self.m_sum[0, 0] / self.m_sum.sum(axis=0)[0]
    
    def sensibility(self):
        return self.m_sum[1, 1] / self.m_sum.sum(axis=0)[1]
    
    def get_values(self):
        return [self.acc_mean(), self.acc_median(), self.acc_min(), self.acc_max(), self.acc_std(), self.specificity(), self.sensibility()]
    
    def print_all(self):
        #print('Confusion Matrix:')
        #cm = pd.DataFrame(data=self.m_sum)
        #display(cm)
        print('Test Accuracy:')
        accs = pd.DataFrame(data=[self.get_values()],
                            columns=['Mean', 'Median', 'Min', 'Max', 'STD', 'Specificity', 'Sensibility'])
        display(HTML(accs.to_html(index=False)))
