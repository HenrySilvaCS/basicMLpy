# # basicMLpy.classification module
import numpy as np
from scipy import linalg
from basicMLpy.utils import check_for_intercept

#CLASSIFICATION#

def probability_k1(row,parameters):
    """
    Calculates Pr(G = 0|X = x).
    Inputs:
        row: array_like
             input array(row vector), represents a row of the matrix of input points.
        parameters: array
             input the column vector of predictors.
    Returns:
        result: float
              outputs Pr(G = 0|X = x).
    """
    result = np.exp(row @ parameters)/(1 + np.exp(row @ parameters))
    return result 

def probability_vector(dataset,parameters):
    """
    Calculates the vector of probabilities. Each element represents the probability of a given x belonging to class k = 0.
    Inputs:
        dataset: array
             input array of input points, already expected to include the intercept. 
        parameters: array
             input the column vector of predictors.
    Returns: 
        p: array
             outputs the p vector of probabilities.
    """
    p = np.zeros((len(dataset),1))
    for i in range(len(dataset)):
        p[i] = probability_k1(dataset[i,:],parameters)
    return p 

def weight_matrix(dataset,parameters):
    """
    Calculates the diagonal matrix of weights, defined by: W[i,i] = Pr(G = 0|X = x_i) * (1 - Pr(G = 0|X = x_i)).
    Inputs:
        dataset: array
             input array of input points, already expected to include the intercept.
        parameters: array
             input the column vector of predictors.
    Outputs:
        w: array
             outputs a diagonal matrix NxN(N being the number of train samples).
    """
    w = np.eye(len(dataset))
    for i in range(len(dataset)):
        w[i,i] = probability_k1(dataset[i,:],parameters) * (1 - probability_k1(dataset[i,:],parameters))
    return w 

def newton_step(dataset,y,n_iter):
    """
    Calculates the newton step for a given array of input points and it's corresponding vector of output points.
    Inputs:
        dataset: array
             input array of input points, already expected to include the intercept.
        y: array
             input array of output points.
        n_iter: int
            Input the number of iterations for the IRLS algorithm. The algorithm is pretty expensive, so I recommend starting with small values(by experience 15 seems to be a good guess) and then start slowly increasing it untill convergence.
    Returns:
        theta: array
            outputs array of predictors/parameters calculated by the algorithm.   
    """
    theta = np.zeros((np.size(dataset,1),1))
    for i in range(n_iter):
        z = dataset @ theta + np.linalg.pinv(weight_matrix(dataset,theta)) @ (y - probability_vector(dataset,theta))
        theta = np.linalg.pinv(dataset.T @ weight_matrix(dataset,theta) @ dataset) @ dataset.T @ weight_matrix(dataset,theta) @ z 
    return theta


def one_vs_all_default(x,y,k,n_iter):
    """
    Fits a one-vs-all classification model on a given dataset of k > 2 classes.
    Inputs:
        x: array
            input array of input points to be used as training set
        y: array
            input array of output points.
        n_iter: int
            Input the number of iterations for the IRLS algorithm. The algorithm is pretty expensive, so I recommend starting with small values(by experience 15 seems to be a good guess) and then start slowly increasing it untill convergence.
    Returns:
        theta: array
            outputs array of predictors/parameters calculated by the algorithm.

    """
    if k <= 2:
        raise ValueError("Number of classes must be bigger than two!")
    else:
        x = check_for_intercept(x)
        prob_index = np.zeros((len(x),k))
        theta = np.zeros((np.size(x,1),k))
        target = y.copy()
        for i in range(k):
            for w in range(len(y)):
                if y[w] == i:
                    target[w] = 1
                else:
                    target[w] = 0

            parameters = newton_step(x,target.reshape((-1,1)),n_iter)
            theta[:,i] = parameters[:,0]
        return theta


class IRLSClassifier:
    """
    Iteratively Reweighted Least Squares algorithmn for classification, that can solve both binary and multiclass problems.
    Methods:
        fit(X,y) -> Performs the IRLS algorithm on the training set(x,y).
        predict(x) -> Predict the class for X.
        get_prob -> Predict the probabilities for X.
        parameters() -> Returns the calculated parameters for the linear model.

    """
    def __init__(self,k,n_iter=15):
        """
        Initialize self.
        Inputs:
            k: int
                input the number k of classes associated with the dataset. 
            n_iter: int
                Input the number of iterations for the IRLS algorithm. The algorithm is pretty expensive, so I recommend starting with small values(by experience 15 seems to be a good guess) and then start slowly increasing it untill convergence; default is set to 15.
        """
        self.k = k 
        if self.k <= 1:
            raise ValueError("Insert a valid number of classes!")
        self.n_iter = n_iter

    def fit(self,x,y):
        """
        Fits the classification model on the dataset.
        Inputs:
            x: array
                input array of input points to be used as training set, without the intercept(raw data).
            y: array
                input array of output points, usually a column vector with same number of rows as x.
        Functionality:
            stores all information from a given function into self.result. this statement will be used later in other functions.            
        """
        self.x = check_for_intercept(x)
        self.y = y 
        if self.k == 2:
            self.parameters = newton_step(self.x,self.y,self.n_iter)
        elif self.k >= 3 : 
            self.parameters = one_vs_all_default(self.x,self.y,self.k,self.n_iter)
    def predict(self,x):
        """
        Gives the prediction made by a certain function based on the input.
        Inputs:
            x: float or array
                input the array of input points or a single input point.
        Returns:
            predictions: float or array
                outputs the prediction made by the classification algorithm.
        """
        x = check_for_intercept(x)
        if self.k == 2:
            predictions = np.round(probability_vector(x,self.parameters))
            return predictions
        else:
            probability_matrix = np.zeros((len(x),self.k))
            predictions = np.zeros((len(x),1))
            for j in range(self.k):
                prob = probability_vector(x,self.parameters[:,j])
                probability_matrix[:,j] = prob[:,0]
            for i in range(len(predictions)):
                predictions[i,0] = np.argmax(probability_matrix[i,:])   
            return predictions 
    def get_prob(self,x):
        """
        Gives the predicted probabilities(without rounding up the results) made by the IRLS algorithm.
        Inputs:
            x: float or array
                input the array of input points or a single input point.
        Returns:
            predictions: float or array
                outputs the prediction(probability) made by the classification algorithm.       
        """
        x = check_for_intercept(x)
        if self.k == 2:
            predictions = probability_vector(x,self.parameters)
            return predictions
        else:
            probability_matrix = np.zeros((len(x),self.k))
            predictions = np.zeros((len(x),1))
            for j in range(self.k):
                prob = probability_vector(x,self.self.parameters[:,j])
                probability_matrix[:,j] = prob[:,0]
            for i in range(len(predictions)):
                predictions[i] = probability_matrix[i,np.argmax(probability_matrix[i,:])]   
            return predict
    def parameters(self):
        """
        Gives the parameters calculated by the classification function.
        Returns:
            self.result[0]: array
                outputs the array of parameters calculated by the classification function.
        """
        return self.parameters

##############
