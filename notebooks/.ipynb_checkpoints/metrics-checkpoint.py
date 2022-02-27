import numpy as np

def squared_error(pred_y, expec_y):
    return (pred_y - expec_y) ** 2

def absolute_error(pred_y, expec_y):
    return (pred_y - expec_y)

def mean_squared_error(pred_y, expec_y):
    return squared_error(pred_y, expec_y).mean()

def mean_absolute_error(pred_y, expec_y):
    return absolute_error.mean()
    
def r2 (pred_y, expec_y):
    var_a = sum([ (y_i - y_t) ** 2 for y_i, y_t in zip(pred_y, expec_y) ])
    var_b = sum([ (y_i - expec_y.mean()) ** 2 for y_i in expec_y ])
        
    return 1 - (var_a/var_b)

def cross_entropy(y_pred, y_expec):
    return (-y_expec*np.log(y_pred) - (1 - y_expec)*np.log(1 - y_pred)).mean()