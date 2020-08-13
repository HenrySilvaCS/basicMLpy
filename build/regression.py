# # basicMLpy.regression module
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import linalg 
def regressionQR(x,y):
    """
    Calculates the predictors for a linear regression model, using QR decomposition.
    Inputs:
        x: array
            input array of input points, with the intercept.
        y: array
            input array of output points, usually a column vector with same number of rows as x.
    Returns:
        theta: array
            outputs the array of predictors for the regression model.
    """
    q, r = np.linalg.qr(x) 
    b = q.T @ y
    theta = linalg.solve_triangular(r,b)
    return theta
def calculate_error(x,y,parameters):
    """
    Calculates the squared error of a given model.
    Inputs:
        x: array
            input array of input points, with the intercept.
        y: array
            input array of output points, usually a column vector with same number of rows as x.
        parameters: array
            input the column vector of predictors.
    Returns:
        errors: array
            outputs a column vector of squared errors.
        errors_sum: float
            outputs the sum of the errors vector.
    """
    prediction = (x @ parameters).reshape((np.size(x,0)),1)
    errors = np.square(prediction - y)
    errors_sum = np.sum(errors)
    return errors, errors_sum
def linear_regression(x,y,tsize):
    """
    Fits a linear regression model on a given dataset.
    Inputs:
        x: array
            input array of input points to be used as training set, without the intercept(raw data).
        y: array
            input array of output points, usually a column vector.
        tsize: float
            Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used on the validation set.
    Returns:
        theta: array
            outputs the array of predictors for the regression model.
        mse: float
            outputs the Mean Squared Error found by the regression model on the validation set. mse is rounded up to 2 decimal cases.
        prediction: array
            outputs the array of predictions over all inputs using the calculated parameters.
        huber_loss: float
            outputs the huber loss for the validation set.
    """
    ones = np.ones((len(x),1))
    x = np.hstack((ones,x))
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = tsize, random_state=5)
    X_test = X_test.reshape((-1,X_train.shape[1]))
    Y_test = Y_test.reshape((-1,1))
    theta = regressionQR(X_train,Y_train)
    error = calculate_error(X_test,Y_test,theta)[1]
    mse = error/np.size(X_test,0)
    prediction = x @ theta
    huber_loss = 0
    pred = X_test @ theta
    delta = np.quantile(Y_test - pred,0.5)
    for i in range(len(X_test)):
        if np.absolute(Y_test[i] - pred[i] ) <= delta:
            huber_loss += np.square(Y_test[i] - pred[i])
        else:
            huber_loss += 2 * delta * np.absolute(Y_test[i] - pred[i]) - delta**2
    return theta, mse, prediction, huber_loss
def basis_expansion(x,y,btype,tsize):
    """
    Executes a basis expansion on the array of inputs and then fits a linear regression model on the dataset.
    Inputs:
        x: array
            input array of input points to be used as training set, without the intercept(raw data).
        y: array
            input array of output points, usually a column vector with same number of rows as x.
        btype: string
            input string that identifies the type of basis expansion; btype can be: None, sqrt, poly.
        tsize: float
            Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used on the validation set.
    Returns:
        theta: array
            outputs the array of predictors for the regression model.
        mse: float
            outputs the Mean Squared Error found by the regression model for the validation set. mse is rounded up to 2 decimal cases.
        prediction: array
            outputs the array of predictions over all inputs using the calculated parameters.
        huber_loss: float
            outputs the huber loss for the validation set.
    *REMEMBER TO APPLY THE SAME BASIS EXPANSION ON THE TEST SET.
    """
    ones = np.ones((np.size(x,0),1))
    x = np.hstack((ones,x))
    x_new = np.zeros((len(x),np.size(x,1)))
    for i in range(len(x)):
        x_new[i,:] = x[i,:]
    if btype == 'sqrt':
        X_train, X_test, Y_train, Y_test = train_test_split(x_new, y, test_size = tsize, random_state=5)
        X_test = X_test.reshape((-1,np.size(X_train,1)))
        Y_test = Y_test.reshape((-1,1))
        for i in range(np.size(x,1)):
            X_train[:,i] = np.sqrt(X_train[:,i])
            X_test[:,i] = np.sqrt(X_test[:,i])
        theta = regressionQR(X_train,Y_train)
        error = calculate_error(X_test,Y_test,theta)[1]
        mse = error/np.size(X_test,0)
        prediction = x @ theta
        pred = X_test @ theta
        huber_loss = 0
        delta = np.quantile(Y_test - pred,0.5)
        for i in range(len(X_test)):
            if np.absolute(Y_test[i] - pred[i] ) <= delta:
                huber_loss += np.square(Y_test[i] - pred[i])
            else:
                huber_loss += 2 * delta * np.absolute(Y_test[i] - pred[i]) - delta**2
        return theta,mse, prediction, huber_loss
    if btype == 'poly':
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = tsize, random_state=5)
        X_test = X_test.reshape((-1,np.size(X_train,1)))
        Y_test = Y_test.reshape((-1,1))
        for i in range(np.size(x,1)):
            X_train[:,i] = (X_train[:,i] **(i)) 
            X_test[:,i] = (X_test[:,i] **(i))
        theta = regressionQR(X_train,Y_train)
        error = calculate_error(X_test,Y_test,theta)[1]
        mse = error/np.size(X_test,0)
        prediction = x @ theta
        pred = X_test @ theta
        huber_loss = 0
        delta = 0.5
        for i in range(len(X_test)):
            if np.absolute(Y_test[i] - pred[i] ) <= delta:
                huber_loss += np.square(Y_test[i] - pred[i])
            else:
                huber_loss += 2 * delta * np.absolute(Y_test[i] - pred[i]) - delta**2
        return theta,mse, prediction, huber_loss
    else:
        print("Insert a valid expansion type")
        return ValueError 
def ridge_regression(x,y,const,tsize):
    """
    Fits a ridge linear regression model on a given dataset.
    Inputs:
        x: array
            input array of input points to be used as training set, with the intercept.
        y: array
            input array of output points, usually a column vector with same number of rows as x.
        const: float
            input the value for the penalizing constant(lambda) used by the ridge algorithm.
        tsize: float
            Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used on the validation set.
    Returns:
        theta: array
            outputs the array of predictors for the regression model.
        mse: float
            outputs the Mean Squared Error found by the regression model. mse is rounded up to 2 decimal cases.
        prediction: array
            outputs the array of predictions over all inputs using the calculated parameters.
        huber_loss: float
            outputs the huber loss for the validation set.
    """
    ones = np.ones((np.size(x,0),1))
    x = np.hstack((ones,x))
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = tsize, random_state=5)
    X_test = X_test.reshape((-1,np.size(X_train,1)))
    Y_test = Y_test.reshape((-1,1))
    identity = np.eye(np.size(X_train,1))
    theta = np.linalg.inv(X_train.T @ X_train + const * identity) @ X_train.T @ Y_train 
    error = calculate_error(X_test,Y_test,theta)[1]
    mse = error/np.size(X_test,0)
    prediction = x @ theta
    pred = X_test @ theta
    huber_loss = 0
    delta = np.quantile(Y_test - pred,0.5)
    for i in range(len(X_test)):
        if np.absolute(Y_test[i] - pred[i] ) <= delta:
            huber_loss += np.square(Y_test[i] - pred[i])
        else:
            huber_loss += 2 * delta * np.absolute(Y_test[i] - pred[i]) - delta**2
    return theta, mse, prediction, huber_loss
def mse_and_huber(prediction,ytest):
    """
    Calculates the Mean Squared Error and the Huber Loss of a given model w.r.t. the test dataset.
    Inputs:
        prediction: array
            input array of predictions made by some algorithm.
        ytest: array
            input array of all points that constitute the output test dataset.
    Returns:
        mse: float
            outputs the Mean Squared Error w.r.t. the test set.
        huber_loss: float
            outputs the Huber Loss w.r.t. the test set.
            
        
"""
    error = np.sum(np.square(ytest - prediction))
    mse = error/len(ytest)
    huber_loss = 0
    delta = 0.5
    for i in range(len(ytest)):
        if np.absolute(ytest[i] - prediction[i] ) <= delta:
            huber_loss += np.square(ytest[i] - prediction[i])
        else:
            huber_loss += 2 * delta * np.absolute(ytest[i] - prediction[i]) - delta**2
    return mse, huber_loss
class LinearRegression:
    """
    Class of two different linear regression models.
    Methods:
        fit(X,y) -> Performs the linear regression algorithm on the training set(x,y).
        predict(x) -> Predict value for X.
        parameters() -> Returns the calculated parameters for the linear model.
        val_error(etype) -> Returns the validation error of the model.
    """
    def __init__(self, reg_type = 'standard',tsize = 0.2):
        """
        Initialize self. Allows the user to choose from two differente types of regression models.
        Inputs:
            reg_type: string
                input string that identifies the type of regression; reg_type can be: 'standard'(standard linear regression) or 'ridge'(ridge regression); default is set to 'standard'.
            tsize: float
                Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used on the validation set; default is set to 0.2.
        """
        self.tsize = tsize
        self.reg_type = reg_type
    def fit(self,x,y,reg_lambda = None):
        """
        Fits one of the regression models on the dataset.
        Inputs:
            x: array
                input array of input points, without the intercept(raw data).
            y: array
                input array of output points, usually a column vector.
            reg_lambda: float
                input value that specifies the regularization constant to be used in the ridge regresssion; default is set to None.
        Functionality:
            stores all information from a given function into self.result. this statement will be used later in other functions.
        """
        self.x = x 
        self.y = y 
        self.reg_lambda = reg_lambda 
        if self.reg_type == 'standard':
            self.result = linear_regression(self.x,self.y,self.tsize)
        elif self.reg_type == 'ridge':
            if reg_lambda == None:
                raise ValueError("Insert a value for the regularization constant.")
            else:
                self.result = ridge_regression(self.x,self.y,self.reg_lambda,self.tsize)
        else:
            raise ValueError("Please insert a valid regression type.")
    def predict(self,x):
        """
        Gives the prediction made by a certain function based on the input.
        Inputs:
            x: float or array
                input the array of input points or a single input point.
        Returns:
            self.predict: float or array
                outputs the prediction made by the classification algorithm
        """
        inputx = x  
        ones = np.ones((len(inputx),1))
        inputx = np.hstack((ones,inputx))
        predict = inputx @ self.result[0]
        return predict
    def parameters(self):
        """
        Gives the parameters calculated by the regression function.
        Returns:
            self.result[0]: array
                outputs the array of parameters calculated by the regression function.
        """
        return self.result[0]
    def val_error(self,etype = 'mse'):
        """
        Gives two different types of error calculated on the validation set,i.e., calculates the expected test error using the validation set.
        Inputs:
            etype: string
                input string that identifies the type of error to be calculated. etype can be: 'mse'(Mean Squared Error) or 'huber'(Huber Loss); default is set to 'mse'.
        Returns:
            self.result[1]: float
                outputs the Mean Squared Error of the model calculated on the validation set.
            self.result[3]: float
                outputs the Huber Loss of the model calculated on the valiation set.
        """
        self.etype = etype
        if self.etype == 'mse':
            return float(self.result[1])
        elif self.etype == 'huber':
            return float(self.result[3])
        else:
            raise ValueError("Please insert a valid error type.")

class BERegression:
    """
    Class of basis expanded regression models, that allow for nonlinear models.
    Methods:
        fit(X,y) -> Performs the linear regression algorithm on the training set(x,y).
        predict(x) -> Predict value for X.
        parameters() -> Returns the calculated parameters for the linear model.
        val_error(etype) -> Returns the validation error of the model.
    """
    def __init__(self,btype,tsize=0.2):
        """
        Initialize self.
        Inputs:
            btype: string
                input string that identifies the type of basis expansion; btype can be: 'sqrt'(square root expanded regression) or 'poly'(polynomial expanded regression).
            tsize: float
                Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used on the validation set; default is set to 0.2.   
        """
        self.btype = btype
        self.tsize = tsize
    def fit(self,x,y):
        """
        Fits  the regression model on the dataset.
        Inputs:
            x: array
                input array of input points, without the intercept(raw data).
            y: array
                input array of output points, usually a column vector.
        Functionality:
            stores all information from a given function into self.result. this statement will be used later in other functions.
        """
        self.x = x 
        self.y = y 
        self.result = basis_expansion(self.x,self.y,self.btype,self.tsize)
    def predict(self,x):
        """
        Gives the prediction made by a certain function based on the input.
        Inputs:
            x: float or array
                input the array of input points or a single input point.
        Returns:
            self.predict: float or array
                outputs the prediction made by the classification algorithm
        """
        inputx = x  
        ones = np.ones((len(inputx),1))
        inputx= np.hstack((ones,inputx)) 
        x_new = np.zeros((inputx.shape))
        for i in range(len(x)):
            x_new[i,:] = inputx[i,:]
        if self.btype == 'sqrt':
            for i in range(len(x_new)):
                x_new[i,:] = np.sqrt(x_new[i,:])
        elif self.btype == 'poly':
            for i in range(np.size(inputx,1)):
                x_new[:,i] = (x_new[:,i] ** i)
        else:
            raise ValueError("Enter a valid type of basis expansion") 
        predict = x_new @ self.result[0]
        return predict
    def parameters(self):
        """
        Gives the parameters calculated by the regression function.
        Returns:
            self.result[0]: array
                outputs the array of parameters calculated by the regression function.
        """
        return self.result[0]
    def val_error(self,etype = 'mse'):
        """
        Gives two different types of error calculated on the validation set,i.e., calculates the expected test error using the validation set.
        Inputs:
            etype: string
                input string that identifies the type of error to be calculated. etype can be: 'mse'(Mean Squared Error) or 'huber'(Huber Loss); default is set to 'mse'.
        Returns:
            self.result[1]: float
                outputs the Mean Squared Error of the model calculated on the validation set.
            self.result[3]: float
                outputs the Huber Loss of the model calculated on the valiation set.
        """
        self.etype = etype
        if self.etype == 'mse':
            return float(self.result[1])
        elif self.etype == 'huber':
            return float(self.result[3])
        else:
            raise ValueError("Please insert a valid error type.")        
#####
