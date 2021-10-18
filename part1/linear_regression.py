import numpy as np

### Functions for you to fill in ###

def closed_form(X, Y, lambda_factor):
    n = np.transpose(X).shape[0]
    theta = np.dot(np.linalg.inv(np.dot(np.transpose(X), X) + lambda_factor * np.identity(n)),
                   np.dot(np.transpose(X), Y))
    return theta
    raise NotImplementedError

### Functions which are already complete, for you to use ###

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
