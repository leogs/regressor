import numpy as np
from metrics import *

class SimpleRegressor:
    
    def __init__(self, intercept=True):
        self.intercept = intercept
        self.betas = None
        self.is_fitted = False

    def predict(self, X, intercept=True):
        X = np.c_[np.ones(X.shape[0]), X] if intercept else X
        
        try:
            pred = np.dot(X, self.betas)
        except NameError:
            raise
        return np.dot(X, self.betas)
    
    def __ordinary_least_square(self, X, y, **kwargs):
        X = np.c_[np.ones(X.shape[0]), X] if self.intercept else X
        
        self.betas = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    def __gradient_descent(self, X, y, learning_rate, steps, seed, verbosity, cost_func='mse', **kwargs):
        np.random.seed(seed)
        
        X = np.c_[np.ones(X.shape[0]), X] if self.intercept else X
        
        minimize = {
            'mse': self.mean_squared_error,
            'mae': self.mean_absolute_error,
            
        }
        
        self.betas = np.random.rand(X.shape[1])
        
        for i in range(steps):
            predicted = self.predict(X, self.betas, intercept=False)
            error = minimize.get(cost_func, lambda: 'Invalid')(predicted, y)
                
            if verbosity != 0 and i % verbosity == 0 :
                print('Step: {} --- Error: {}'.format(i, error))
            
            error = self.absolute_error(predicted,  y)
            gradient = 2*np.dot(X.T, error) / len(X)
            self.betas -= learning_rate*gradient

    def fit(self, X, y, mode='ols', learning_rate=0.01, steps=10000, seed=0, verbosity=0, cost_func='mse'):
        modes = {
            'ols': self.__ordinary_least_square,
            'gradient_descent': self.__gradient_descent
        }
        
        return modes.get(mode, lambda: 'Invalid')(X=X,
                                                y=y,
                                                learning_rate=learning_rate,
                                                steps=steps,
                                                seed=seed,
                                                verbosity=verbosity,
                                                cost_func=cost_func)
