import numpy as np
import warnings
from metrics import *

class SimpleLinearRegressor:
    
    def __init__(self, intercept=True):
        self.intercept = intercept
        self.betas = None

    def predict(self, X, intercept=True):
        X = np.c_[np.ones(X.shape[0]), X] if intercept else X
        
        try:
            pred = np.dot(X, self.betas)
        except TypeError as e:
            raise Exception('Model is not fitted. Call fit before predict').with_traceback(e.__traceback__)
        return pred
    
    def __ordinary_least_square(self, X, y, **kwargs):
        self.betas = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        self.is_fitted = True

    def __gradient_descent(self, X, y, learning_rate, steps, seed, verbosity, cost_func='mse', **kwargs):
        np.random.seed(seed)
        
        minimize = {
            'mse': mean_squared_error,
            'mae': mean_absolute_error,
            
        }
        
        self.betas = np.random.rand(X.shape[1])
        
        for i in range(steps):
            predicted = self.predict(X, intercept=False)
            
            try:
                error = minimize.get(cost_func, lambda: 'Invalid')(predicted, y)
            except ValueError as e:
                raise Exception('X and y have different lenght')
                
            if verbosity != 0 and i % verbosity == 0 :
                print('Step: {} --- Error: {}'.format(i, error))
            
            error = absolute_error(predicted,  y)
            gradient = 2*np.dot(X.T, error) / len(X)
            self.betas -= learning_rate*gradient


    def fit(self, X, y, mode='ols', learning_rate=0.01, steps=10000, seed=0, verbosity=0, cost_func='mse'):
        modes = {
            'ols': self.__ordinary_least_square,
            'gradient_descent': self.__gradient_descent
        }
        
        try:
            X = np.c_[np.ones(X.shape[0]), X] if self.intercept else X
        except AttributeError as e:
            raise Exception('X must a be an numpy array').with_traceback(e.__traceback__)
        
        try:
            modes.get(mode)(X=X,
                            y=y,
                            learning_rate=learning_rate,
                            steps=steps,
                            seed=seed,
                            verbosity=verbosity,
                            cost_func=cost_func)
            
        except TypeError as e:
            modes.get('ols')(X=X,
                            y=y,
                            learning_rate=learning_rate,
                            steps=steps,
                            seed=seed,
                            verbosity=verbosity,
                            cost_func=cost_func)
            warnings.warn('Invalid parameter: {}. Using ols.'.format(mode))
        except Exception as e:
            raise e
