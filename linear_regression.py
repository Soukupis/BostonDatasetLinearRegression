class linear_regression:
    def __init__(self):
       pass 
    def compute_slope( x, y, x_mean, y_mean):
        frac1 = sum([(x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x))])
        frac2 = sum([(x[i] - x_mean)**2 for i in range(len(x))])
        slope = frac1 / frac2
        return slope

    def compute_intercept(x_mean, y_mean, slope):
        intercept = y_mean - slope * x_mean
        return intercept

    def compute_regression(x, slope, intercept):
        regression_line = [slope * x[i] + intercept for i in range(len(x))]
        return regression_line
    
    def compute_r2(y, y_mean, regression_line):
        frac1 = sum([(y[i] - regression_line[i])**2 for i in range(len(y))])
        frac2 = sum([(y[i] - y_mean)**2 for i in range(len(y))])
        r2 = 1 - frac1 / frac2
        return r2

