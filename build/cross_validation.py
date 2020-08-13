# # basicMLpy.cross_validation module
import numpy as np
from basicMLpy import regression as bpr 
from basicMLpy import classification as bpc
from basicMLpy import ensemble as bpe 
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
    prediction_copy = prediction.reshape((-1,1))
    y_copy = y.reshape((-1,1))
    sum_squares = np.sum(np.square(prediction_copy - y_copy))
    return sum_squares/len(y_copy)
def k_folds_standard(x,n_folds):
    """
    Divides the input array into n_folds different folds/subsets.
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
    Methods:
        fit(X,y) -> Performs the cross validation algorithm on the training set(x,y).
        cv_scores() -> Gives the cross validation scores for the training set.
        expected_generalization_error() -> Gives the predicted generalization(out of sample) test error.
        get_best_parameters() -> Returns the parameters that correspond to the model with lowest cv score; only works for the models of basicMLpy.regression and basicMLpy.classification.
    """
    def __init__(self,n_folds,function,**kwargs):
        """
        Initialize self.
        Inputs:
            n_folds: int
                input number of folds to be created. must be bigger than two.
            function: string
                input string that identifies the function to use in the cross-validation algorithm; function can be: 'LinearRegression', 'RidgeRegression', 'BERegression', 'IRLSClassifier'. see basicMLpy documentation for more information on the functions.
            **kwargs: dict 
                input specific parameters for the respective function.
        """
        possible_func = ['LinearRegression','RidgeRegression','BERegression','IRLSClassifier','RandomForestRegressor','RandomForestClassifier','AdaBoostClassifier','GBRegressor']
        assert n_folds >= 1
        assert function in possible_func
        if function == 'LinearRegression':
            self.n_folds = n_folds
            self.function = function 
            self.tsize = kwargs.get('tsize')
        elif function == 'RidgeRegression':
            self.reg_lambda = kwargs.get('reg_lambda')
            assert self.reg_lambda != None
            self.n_folds = n_folds
            self.function = function 
            self.tsize = kwargs.get('tsize')
            self.reg_type = kwargs.get('reg_type')
            assert self.reg_type == 'ridge'
        elif function == 'BERegression':
            self.n_folds = n_folds
            self.function = function 
            self.btype = kwargs.get('btype')
            possible_btype = ['poly','sqrt']
            assert self.btype != None and self.btype in possible_btype
            self.tsize = kwargs.get('tsize')
        elif function == 'IRLSClassifier':
            self.k = kwargs.get('k')
            assert self.k >= 1
            self.n_folds = n_folds 
            self.function = function 
            self.tsize = kwargs.get('tsize')
            self.n_iter = kwargs.get('n_iter')
            assert self.n_iter != None
        elif function == 'RandomForestRegressor':
            self.n_folds = n_folds
            self.function = function 
            self.n_estimators = kwargs.get('n_estimators')
            self.max_features = kwargs.get('max_features')
            self.max_depth = kwargs.get('max_depth')
            self.criterion = kwargs.get('criterion')
            self.random_state = kwargs.get('random_state')
        elif function == 'RandomForestClassifier':
            self.n_folds = n_folds
            self.function = function 
            self.n_estimators = kwargs.get('n_estimators')
            self.n_classes = kwargs.get('n_classes')
            self.max_features = kwargs.get('max_features')
            self.max_depth = kwargs.get('max_depth')
            self.criterion = kwargs.get('criterion')
            self.random_state = kwargs.get('random_state')
        elif function == 'AdaBoostClassifier':
            self.n_folds = n_folds
            self.function = function 
            self.n_estimators = kwargs.get('n_estimators')
        else:
            self.n_folds = n_folds
            self.function = function 
            self.n_estimators = kwargs.get('n_estimators')
            self.loss_func = kwargs.get('loss_func')
            self.max_depth = kwargs.get('max_depth')
            self.random_state = kwargs.get('random_state')
            self.max_features = kwargs.get('max_features')


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
                func = bpc.IRLSClassifier(k=self.k,tsize=self.tsize,n_iter=self.n_iter)
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
                func = bpr.LinearRegression(tsize=self.tsize)
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
                func = bpr.LinearRegression(reg_type = self.reg_type,tsize=self.tsize)
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
                func = bpr.BERegression(btype = self.btype,tsize=self.tsize)
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
        elif self.function == 'RandomForestClassifier':
            for i in range(self.n_folds):
                func = bpe.RandomForestClassifier(n_estimators =self.n_estimators,n_classes = self.n_classes, max_depth = self.max_depth, criterion = self.criterion, random_state = self.random_state, max_features = self.max_features)
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
        elif self.function == 'RandomForestRegressor':
            for i in range(self.n_folds):
                func = bpe.RandomForestRegressor(n_estimators =self.n_estimators, max_depth = self.max_depth, criterion = self.criterion, random_state = self.random_state, max_features = self.max_features)
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
        elif self.function == 'AdaBoostClassifier':
            for i in range(self.n_folds):
                func = bpe.AdaBoostClassifier(n_estimators = self.n_estimators)
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
        else:
            for i in range(self.n_folds):
                func =bpe.GBRegressor(n_estimators =self.n_estimators,loss_func = self.loss_func, max_depth = self.max_depth,random_state = self.random_state,max_features = self.max_features )
                input_train_set = np.delete(folds_input,i,0)
                input_train_set = np.concatenate(input_train_set)
                input_test_set = np.array(folds_input[i])
                output_train_set = np.delete(folds_output,i,0)
                output_train_set = np.concatenate(output_train_set)
                output_test_set = folds_output[i]
                output_train_set = output_train_set.reshape((-1,))
                func.fit(input_train_set,output_train_set)
                predictions = func.predict(input_test_set)
                accuracy = mse(predictions,output_test_set)
                self.scores[i] = accuracy            
        self.parameters = np.array(self.parameters)
    def cv_scores(self):
        """
        Gives the calculated cross-validation scores for the dataset.
        Returns:
            self.scores: array
                outputs the array of cross-validation scores calculated by the algorithm. these scores represent the error of each iteration of the cross-validation.
        """
        return self.scores   
    def expected_generalization_error(self):
        """
        Calculates the expected test error of the model, that by definition is the average of the sum of the cross-validation error found by the algorithm.
        Returns:
            self.error: float
                Outputs the expected test error of the model.
        """
        error = sum(self.scores)/self.n_folds
        return error
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