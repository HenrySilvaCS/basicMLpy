# # basicMLpy.utils module
import numpy as np

def check_for_intercept(x):
    """
    Checks if a given array has an intercept column,i.e. a column of ones. Adds the intercept if the array doesn't have it.
    Inputs:
        x: array
            input array of input points.
    Returns:
        x: array
            outputs the modified array of input points.

    """
    if(len(x.shape) != 1):
        if(not (x[:,0] == 1).all()):
            print("Array of inputs was passed without intercept.\nAdding intercept...")
            ones = np.ones((len(x),1))
            x = np.hstack((ones,x))
            return x 
        else:
            return x 
    else:
        print("Array of inputs was passed without intercept. Adding intercept...")
        ones = np.ones((len(x)))
        x = np.vstack((ones,x)).T
        return x 

def residuals(y,prediction,loss_func):
    """
    Calcultes the generalized(or partial) residuals between the prediction and the output. The residuals are defined to be the negative value of the gradient of the loss function w.r.t. prediction.
    Inputs:
        y: array
            input the array of outputs.
        prediction: array
            input the array of predictions.
        loss_func: string
            input the string that identifies the loss function to use when calculating the residuals; loss_func can be 'mse'(Mean Squared Error), 'mae'(Mean Absolute error).
    Returns:
        gradient: ndarray
            outputs the residuals between the prediction and the output.

    """
    possible_loss_funcs = ['mse','mae']
    assert loss_func in possible_loss_funcs

    if loss_func == possible_loss_funcs[0]:
        return (y - prediction)
    else:
        return (2*(y - prediction) - 1).reshape((-1,1))

def optimal_gamma(y,prediction,loss_func):
    """
    Calculates the optimal value for the gamma constant based on a certain loss_func.
    Inputs:
        y: array
            input the array of outputs/targets.
        prediction: array
            input the array of predictions.
        loss_func: string
            input the string that identifies the loss function to be used; loss_func can be: 'mse'(Mean Squared Error) or 'mae'(Mean Absolute Error)
    Returns:
        np.median(res) or np.mean(res): float or ndarray
            outputs the optimal value for the gamma parameter given the prediction and outputs.
    """
    assert loss_func in ["mse","mae"]

    if loss_func == 'mae':
        res = y - prediction
        return np.median(res)  

    elif loss_func == 'mse':
        res = y - prediction
        return np.mean(res)

def split_indices(x,n_folds):
    """
    Splits the indices of a given array into n folds.
    Inputs:
        x: array
            input the array of input points.
        n_folds: int
            input the number of folds to create.
    Returns:
        fold_idxs: list of lists of int
            outputs a list of folds. each fold is a list of indices.
    """
    try:
        assert len(x) % n_folds == 0
    except:
        raise ValueError("Number of samples must be divisible by the number of folds!")

    fold_idxs = []
    indices = list(range(len(x)))
    np.random.shuffle(indices)
    fold_size = int(len(x)/n_folds)

    for i in range(n_folds):
        if(i == 0):
            fold_idxs.append(indices[:((i+1) * fold_size)])
        elif((i != n_folds - 1)):
            fold_idxs.append(indices[((i) * fold_size):((i+1) * fold_size) ])
        else:
            fold_idxs.append(indices[((i) * fold_size):])

    return fold_idxs
    
def euclidean_distance(x1, x2):
    """
    Calculates the euclidian distance between two points.
    Inputs:
        x1: array_like
            input a point(row vector or just a float/int number) to be used in the calculation.
        x2: array_like
            input a point(row vector or just a float/int number) to be used in the calculation.
    Returns: 
        np.sqrt(distance): float
            outputs the euclidean distance between x1 and x2.
    """
    distance = 0
    for i in range(len(x1)-1):
        distance += (x1[i] - x2[i])**2
    return np.sqrt(distance)


def z_normalize(X):
    """
    Performs a z-normalization on a given input.
    Inputs:
        X: ndarray
            input the array on which to perform the z-normalization. this assumes the array has the following format: rows = samples, columns = features. pass X.T otherwise.

    Returns: 
        z_normalized array:
            outputs the z-normalized array, with each of the columns with mean = 0 and standard deviation = 1.
    """
    sample_mean = np.mean(X,axis=0)
    X_centered = X - sample_mean
    variance = np.var(X,ddof=1,axis=0)
    std_dev = np.sqrt(variance)
    return X_centered/std_dev

