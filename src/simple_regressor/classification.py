import numpy as np
from metrics import *

class SimpleLogisticRegression:

    def __init__(self):
        self.is_fitted = False
        self.betas = None
        self.is_fitted = False

    def predict_proba(self, X, intercept=True):
        X = np.c_[np.ones(X.shape[0]), X] if intercept else X
        
        z = np.dot(X, self.betas)
        return 1 / (1 + np.exp(-z))

    def predict(self, y, threshold = 0.5):
        return (self.predict_proba(y, self.betas) > 0.5)*1

    def __gradient_descent(self, X, y, learning_rate, steps, seed, verbosity, cost_func='cross_entropy', **kwargs):
        np.random.seed(seed)
        
        X = np.c_[np.ones(X.shape[0]), X]
        
        minimize = {
            'cross_entropy': cross_entropy
        }
        
        self.betas = np.random.rand(X.shape[1])
        
        for i in range(steps):
            predicted = self.predict_proba(X, self.betas, intercept=False)
            error = minimize.get(cost_func, lambda: 'Invalid')(predicted, y)
                
            if verbosity != 0 and i % verbosity == 0 :
                print('Step: {} --- Error: {}'.format(i, error))
            
            error = predicted - y
            gradient = np.dot(X.T, error)/(len(X))
            self.betas -= learning_rate*gradient
        
        self.is_fitted = True
        
    
    def fit(self, X, y, mode='gradient_descent', learning_rate=0.01, steps=10000, seed=0, verbosity=0, cost_func='cross_entropy'):
        modes = {
            'gradient_descent': self.gradient_descent
        }
        
        return modes.get(mode, lambda: 'Invalid')(X=X,
                                                y=y,
                                                learning_rate=learning_rate,
                                                steps=steps,
                                                seed=seed,
                                                verbosity=verbosity,
                                                cost_func=cost_func)