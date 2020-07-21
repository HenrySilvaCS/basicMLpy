# # basicMLpy.nearest_neighbors module
import numpy
from sklearn.model_selection import train_test_split
from scipy import linalg 
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
            self.predictions: array
                outputs the array of predictions.
        """
        self.input = x 
        self.output = y.reshape((-1,1))
        self.data = np.hstack((self.input,self.output))
        self.predictions = np.zeros((len(self.output)))
        for i in range(len(self.predictions)):
            self.predictions[i] = make_prediction(self.data,i,self.n_neighbors,self.weights)
        return self.predictions 
    def kneighbors(self,row_num,n_neighbors):
        """
        Gets the k-nearest neighbors to a certain point.
        Inputs:
            row_num: int
                input the index(row number) of the desired point to calculate the k nearest neighbors.
            n_neighbors: int
                input the number of neighbors to be calculated.     
        """
        self.row = row_num
        self.k = n_neighbors 
        self.kneighbors = get_neighbors(self.input,self.row,self.k)
        return self.kneighbors