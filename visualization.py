import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import *
from sklearn.datasets import load_boston

dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

np.random.seed(42)


x = dataset.data[:, 6]
y = dataset.target

indices = np.random.permutation(len(x))
test_size =100

x_train = x[indices[:-test_size]]
y_train = y[indices[:-test_size]]

x_test = x[indices[-test_size:]]
y_test = y[indices[-test_size:]]


def train_model(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    m = linear_regression.compute_slope(x, y, x_mean, y_mean)
    b = linear_regression.compute_intercept(x_mean, y_mean, m)
    print("Training is finished")
    print("With Params m: %s and b: %d" % (m, b))
    regression_line = linear_regression.compute_regression(x, m, b)
    r2 = linear_regression.compute_r2(y, y_mean, regression_line)
    print("With R2: %s" % r2)
    return m, b
def test_model(x, y, m, b):
    y_mean = np.mean(y)
    regression_line = linear_regression.compute_regression(x, m, b)
    r2 = linear_regression.compute_r2(y, y_mean, regression_line)
    print("Finished testing")
    print("With R2: %s" % r2)
    return r2

m, b = train_model(x_train, y_train)
r2_test = test_model(x_test, y_test, m, b)

LB = int(np.floor(np.min(x))) -5
UB = int(np.floor(np.max(x))) +5

line = [m * i + b for i in [LB, UB]]

plt.scatter(x_test, y_test, color="blue")
plt.plot([LB, UB], line, color="red")
plt.show()
