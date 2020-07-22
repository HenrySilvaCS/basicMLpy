# # basicMLpy.cross_validation module
import numpy as np
from basicMLpy import regression as bpr 
from basicMLpy import classification as bpc
def error(prediction,y):
    """
    Calculates the error(misclassification) w.r.t. the outputs.
    Inputs:
        prediction: array
            input array of predictions.
        y: array
            input array of output points.
    Returns:
        (counter/len(y)) * 100: float
            outputs the error of the predictions.
    """
    counter = 0
    for i in range(len(y)):
        if prediction[i] == y[i]:
            counter == counter
        else: 
            counter +=1
    return (counter/len(y)) * 100
def mse(prediction,y):
    """
    Calculates the MSE(Mean Squared Error) w.r.t. the outputs.
    Inputs:
        prediction: array
            input array of predictions.
        y: array
            input array of output points.
    Returns:
        sum_squares/len(y): float
            outputs the MSE of the predictions.
    """
    sum_squares = sum(np.square(prediction - y))
    return sum_squares/len(y)
def k_folds_standard(x,n_folds):
    """
    Inputs:
        x: array
            input array of points to be splitted.
        n_folds: int
            input number of folds to be created. must be bigger than two.
    """
    dataset_split = np.array(list(np.vsplit(x,n_folds)))
    return dataset_split
class CrossValidation:
    """
    Class that performs the cross validation given a certain function.
    """
    def __init__(self,n_folds,function,k=None,reg_lambda=None,btype=None):
        """
        Initialize self.
        Inputs:
            n_folds: int
                input number of folds to be created. must be bigger than two.
            function: string
                input string that identifies the function to use in the cross-validation algorithm; function can be: 'LinearRegression', 'RidgeRegression', 'BERegression', 'IRLSClassifier'. see basicMLpy documentation for more information on the functions.
            k: int
                input the number k of classes associated with the dataset; default is set to None. specify if using 'IRLSClassifier'. 
            reg_lambda = float
                input value that specifies the regularization constant to be used in the ridge regresssion; default is set to None. specify if using 'RidgeRegression'.
            btype: string
                input string that identifies the type of basis expansion; btype can be: 'sqrt'(square root expanded regression) or 'poly'(polynomial expanded regression); default is set to None. specify if using 'BERegression'.
        """
        if k == None:
            if function == 'LinearRegression':
                self.n_folds = n_folds
                self.function = function 
            elif function == 'RidgeRegression':
                self.reg_lambda = reg_lambda
                self.n_folds = n_folds
                self.function = function 
                if reg_lambda == None:
                    raise ValueError('Insert a value for the regularization constant')
            elif function == 'BERegression':
                self.n_folds = n_folds
                self.function = function 
                self.btype = btype
                if btype == None:
                    raise ValueError('Insert a type of basis expansion')
        elif k >= 2:
            self.n_folds = n_folds
            self.function = function
            self.k = k 
        else:
            raise ValueError('Invalid value for k, it must be an integer bigger than two')
        if self.n_folds  <= 1:
            raise ValueError("Invalid value for n_folds, it must be bigger than one")
        if function == 'IRLSClassifier' and k == None:
            raise ValueError("Insert a value for k")
    def fit(self,x,y):
        """
        Performs the cross-validation on a given dataset.
        Inputs:
            x: array
                input array of input points, without the intercept(raw data).
            y: array
                input array of output points.
        Functionality:
            stores information about the scores in self.scores and information about the parameters in self.parameters.
        """
        self.input = x
        self.output = y 
        self.output = self.output.reshape((-1,1))
        folds_input = k_folds_standard(self.input,self.n_folds)
        folds_output = k_folds_standard(self.output,self.n_folds)
        self.scores = np.zeros((self.n_folds,))
        self.parameters = list()
        if self.function == 'IRLSClassifier':
            for i in range(self.n_folds):
                func = bpc.IRLSClassifier(k=self.k)
                input_train_set = np.delete(folds_input,i,0)
                input_train_set = np.concatenate(input_train_set)
                input_test_set = np.array(folds_input[i])
                output_train_set = np.delete(folds_output,i,0)
                output_train_set = np.concatenate(output_train_set)
                output_test_set = folds_output[i]
                func.fit(input_train_set,output_train_set)
                predictions = func.predict(input_test_set)
                accuracy = error(predictions,output_test_set)
                self.scores[i] = accuracy 
                self.parameters.append(func.parameters())
        elif self.function == 'LinearRegression':
            for i in range(self.n_folds):
                func = bpr.LinearRegression()
                input_train_set = np.delete(folds_input,i,0)
                input_train_set = np.concatenate(input_train_set)
                input_test_set = np.array(folds_input[i])
                output_train_set = np.delete(folds_output,i,0)
                output_train_set = np.concatenate(output_train_set)
                output_test_set = folds_output[i]
                func.fit(input_train_set,output_train_set)
                predictions = func.predict(input_test_set)
                accuracy = mse(predictions,output_test_set)
                self.scores[i] = accuracy 
                self.parameters.append(func.parameters())
        elif self.function == 'RidgeRegression':
            for i in range(self.n_folds):
                func = bpr.LinearRegression(reg_type = 'ridge')
                input_train_set = np.delete(folds_input,i,0)
                input_train_set = np.concatenate(input_train_set)
                input_test_set = np.array(folds_input[i])
                output_train_set = np.delete(folds_output,i,0)
                output_train_set = np.concatenate(output_train_set)
                output_test_set = folds_output[i]
                func.fit(input_train_set,output_train_set,self.reg_lambda)
                predictions = func.predict(input_test_set)
                accuracy = mse(predictions,output_test_set)
                self.scores[i] = accuracy 
                self.parameters.append(func.parameters())
        elif self.function == 'BERegression':
            for i in range(self.n_folds):
                func = bpr.BERegression(btype = self.btype)
                input_train_set = np.delete(folds_input,i,0)
                input_train_set = np.concatenate(input_train_set)
                input_test_set = np.array(folds_input[i])
                output_train_set = np.delete(folds_output,i,0)
                output_train_set = np.concatenate(output_train_set)
                output_test_set = folds_output[i]
                func.fit(input_train_set,output_train_set)
                predictions = func.predict(input_test_set)
                accuracy = mse(predictions,output_test_set)
                self.scores[i] = accuracy 
                self.parameters.append(func.parameters()) 
        else:
            raise ValueError('Insert a valid function')               
        self.parameters = np.array(self.parameters)
    def cv_scores(self):
        """
        Gives the calculated cross-validation scores for the dataset.
        Returns:
            self.scores: array
                outputs the array of cross-validation scores calculated by the algorithm. these scores represent the error of each iteration of the cross-validation.
        """
        return self.scores   
    def expected_test_error(self):
        """
        Calculates the expected test error of the model, that by definition is the average of the sum of the cross-validation error found by the algorithm.
        Returns:
            self.error: float
                Outputs the expected test error of the model.
        """
        self.error = sum(self.scores)/self.n_folds
        return self.error
    def get_best_parameters(self):
        """
        Gives the best set of parameters based on the model that yielded the smallest cross-validation score.
        Returns:
            self.parameters[index]: array_like
                outputs the array of the parameters that yielded the lowest cross-validation error.
        """
        error = sum(self.scores)/self.n_folds
        index = np.argmin(error)
        return self.parameters[index]
###

