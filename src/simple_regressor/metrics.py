
def squared_error(self, pred_y, expec_y):
    return (pred_y - expec_y) ** 2

def absolute_error(self, pred_y, expec_y):
    return (pred_y - expec_y)

def mean_squared_error(self, pred_y, expec_y):
    return self.squared_error(pred_y, expec_y).mean()

def mean_absolute_error(self, pred_y, expec_y):
    return self.absolute_error.mean()
    
def r2 (self, pred_y, expec_y):
    var_a = sum([ (y_i - y_t) ** 2 for y_i, y_t in zip(pred_y, expec_y) ])
    var_b = sum([ (y_i - expec_y.mean()) ** 2 for y_i in expec_y ])
        
    return 1 - (var_a/var_b)