# # basicMLpy.nearest_neighbors module
import numpy as np
from scipy import linalg 
from basicMLpy.utils import euclidean_distance
def get_neighbors(x, row_num, n_neighbors):
    """
    Gets the K nearest neighbors relative to a point in the dataset.
    Inputs:
        x: array
            input the array of input points.
        row_num: int
            input the index(row number) of the desired point to calculate the k nearest neighbors.
        n_neighbors: int
            input the number of neighbors to be calculated.
    Returns:
        neighbors: array
            outputs the array of the k nearest neighbors to the inputed point.
    """
    distances = list()
    for row in x:
        dist = euclidean_distance(x[row_num,:], row)
        distances.append((row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(n_neighbors):
        neighbors.append(distances[i][0])
    return np.array(neighbors)
def make_prediction(x,row_num,n_neighbors,weights):
    """
    Predicts the output value based on the k nearest neighbors relative to an input point.
    Inputs:
        x: array
            input the array of input points.
        row_num: int
            input the index(row number) of the desired point to make the prediction.
        n_neighbors: int
            input the number of neighbors to be calculated.
        weights: string
            input the type of weight to be used in the calculation. weights can be: 'uniform'(uniform weights,i.e, all points on the neighborhood are weighted equally) or 'distance'(weight points by the inverse of their distance).
    Returns:
        predictions: float
            outputs the prediction calculated based on the input point(row_num).

    """
    neighbors = get_neighbors(x,row_num,n_neighbors)
    outputs = [points[neighbors.shape[1] - 1] for points in neighbors]
    if weights == 'uniform':
        predictions = np.round(sum(outputs)/n_neighbors)
        return predictions
    elif weights == 'distance':
        for i in range(len(outputs)):
            outputs[i] = outputs[i]/(i+1)
        predictions = np.round(sum(outputs)/n_neighbors)
        return predictions
    else:
        raise ValueError("Please insert a valid weight")
class NearestNeighbors:
    """
    Class of the K-Nearest Neighbors algorithm. 
    Fits both regression and classification problems.
    This algorithm performs a brute force search, meaning that it performs poorly on large datasets, since it scales according to O[DN^2], where N is the number of samples and D is the number of dimensions.
    Methods:
        predict(x,y) -> Predict value for X.
        kneighbors(row_num,n_neighbors) -> Gets the k-nearest neighbors from x.
    """
    def __init__(self,n_neighbors = 5,weights = 'uniform'):
        """
        Initial parameters.
        Inputs:
            n_neighbors: int
                input the number of neighbors to be calculated; default is set to 5.
            weights: string
                input the type of weight to be used in the calculation. weights can be: 'uniform'(uniform weights,i.e, all points on the neighborhood are weighted equally) or 'distance'(weight points by the inverse of their distance); default is set to 'uniform'.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
    def predict(self,x,y):
        """
        Makes a prediction based on a given dataset.
        Inputs:
            x: array
                input the array of input points.
            y: array
                input the array of output points.
        Returns:
            predictions: array
                outputs the array of predictions.
        """
        self.input = x 
        self.output = y.reshape((-1,1))
        self.data = np.hstack((self.input,self.output))
        predictions = np.zeros((len(self.output)))
        for i in range(len(predictions)):
            predictions[i] = make_prediction(self.data,i,self.n_neighbors,self.weights)
        return predictions 
    def kneighbors(self,row_num,n_neighbors):
        """
        Gets the k-nearest neighbors to a certain point.
        Inputs:
            row_num: int
                input the index(row number) of the desired point to calculate the k nearest neighbors.
            n_neighbors: int
                input the number of neighbors to be calculated.     
        """ 
        kneighbors = get_neighbors(self.input,row_num,n_neighbors)
        return kneighbors