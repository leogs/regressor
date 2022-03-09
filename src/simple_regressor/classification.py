import numpy as np
from metrics import *

class SimpleLogisticRegression:

    def __init__(self):
        self.betas = None

    def predict_proba(self, X, intercept=True):
        X = np.c_[np.ones(X.shape[0]), X] if intercept else X
        
        try:
            z = np.dot(X, self.betas)
        except TypeError as e:
            raise Exception('Model is not fitted. Call fit before predict').with_traceback(e.__traceback__)
        return 1 / (1 + np.exp(-z))

    def predict(self, X, threshold = 0.5):
        return np.array([1 if e > threshold else 0 for e in self.predict_proba(X)])

    def __gradient_descent(self, X, y, learning_rate, steps, seed, verbosity, cost_func='cross_entropy', **kwargs):
        np.random.seed(seed)
        
        minimize = {
            'cross_entropy': cross_entropy
        }
        
        
        self.betas = np.random.rand(X.shape[1])
        
        for i in range(steps):
            try:
                predicted = self.predict_proba(X, intercept=False)
            except TypeError as e:
                raise Exception('Model is not fitted. Call fit before predict').with_traceback(e.__traceback__)
            
            try:
                error = minimize.get(cost_func, lambda: 'Invalid')(predicted, y)
            except ValueError as e:
                raise Exception('X and y have different lenght')
                
            if verbosity != 0 and i % verbosity == 0 :
                print('Step: {} --- Error: {}'.format(i, error))
            
            error = predicted - y
            gradient = np.dot(X.T, error)/(len(X))
            self.betas -= learning_rate*gradient
        
        self.is_fitted = True
        
    
    def fit(self, X, y, mode='gradient_descent', learning_rate=0.01, steps=10000, seed=0, verbosity=0, cost_func='cross_entropy'):
        modes = {
            'gradient_descent': self.__gradient_descent
        }
        
        try:
            X = np.c_[np.ones(X.shape[0]), X]
        except AttributeError as e:
            raise Exception('X must a be a numpy array').with_traceback(e.__traceback__)
        
        try:
            modes.get(mode, lambda: 'Invalid')(X=X,
                                                y=y,
                                                learning_rate=learning_rate,
                                                steps=steps,
                                                seed=seed,
                                                verbosity=verbosity,
                                                cost_func=cost_func)
        except TypeError as e:
            Exception('Invalid parameter type').with_traceback(e.__traceback__)
        except Exception as e:
            raise e