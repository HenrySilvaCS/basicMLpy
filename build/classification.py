# # basicMLpy.classification module
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import linalg
#CLASSIFICATION#
def probability_k1(row,parameters):
    """
    Calculates Pr(G = 0|X = x).
    Inputs:
        row: array_like
             input array(row vector), represents a row of the matrix of input points(usually represented by X).
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
             outputs the p vector of probabilities
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
             input array of output points, usually a column vector.
    """
    theta = np.zeros((np.size(dataset,1),1))
    for i in range(n_iter):
        z = dataset @ theta + np.linalg.pinv(weight_matrix(dataset,theta)) @ (y - probability_vector(dataset,theta))
        theta = np.linalg.pinv(dataset.T @ weight_matrix(dataset,theta) @ dataset) @ dataset.T @ weight_matrix(dataset,theta) @ z 
    return theta
def binary_classification_default(x,y,tsize):
    """
    Fits a binary classification model on a given dataset of k = 2 classes.
    Splits the dataset(x,y) into training and validation sets, using the train_test_split function from sklearn. Default test_size = 0.2.
    Inputs:
        x: array
            input array of input points to be used as training set, without the intercept(raw data).
        y: array
            input array of output points, usually a column vector with same number of rows as x.
        tsize: float
            Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used on the validation set.
    Returns:
        theta: array
            outputs array of predictors/parameters calculated by the algorithm.
        accuracy: float
            outputs the  percentual accuracy of the model, counting each misclassification on the validation set and calculating the final score.    
        exp_loss: float
            outputs the  exponential loss of the model w.r.t. the validation set.
        prediction_final: array
            outputs the array of predictions over all inputs using the calculated parameters.    
    """
    ones = np.ones((len(x),1))
    x = np.hstack((ones,x))
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = tsize, random_state=5)
    Y_train = Y_train.reshape((len(Y_train),1))
    theta = newton_step(X_train,Y_train,15)
    prediction = probability_vector(X_test,theta)
    p = np.round(prediction)
    counter = 0
    for i in range(len(prediction)):
        if np.absolute(p[i] - Y_test[i]) == 0:
            counter = counter 
        else:
            counter += 1
    accuracy = np.round(((np.size(Y_test,0) - counter)/np.size(Y_test,0)) * 100)
    exp_loss = 0
    for i in range(len(prediction)):
        exp_loss += np.exp(-1 * Y_test[i] * prediction[i])
        prediction_final = probability_vector(x,theta)
    return theta, accuracy, float(exp_loss), np.round(prediction_final)
def one_vs_all_default(x,y,k,tsize):
    """
    Fits a one-vs-all classification model on a given dataset of k > 2 classes.
    Splits the dataset(x,y) into training and validation sets, using the train_test_split function from sklearn. Default test_size = 0.2.
    Inputs:
        x: array
            input array of input points to be used as training set, without the intercept(raw data).
        y: array
            input array of output points, usually a column vector with same number of rows as x.
        tsize: float
            Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used on the validation set.
    Returns:
        theta: array
            outputs array of predictors/parameters calculated by the algorithm.
        accuracy: float
            outputs the approximate percentual accuracy of the model, counting each misclassification on the validation set and calculating the final score.
        exp_loss: float
            outputs the approximate exponential loss of the model on the validation set.
        result_final: array
            outputs the array of predictions over all inputs using the calculated parameters.
    """
    if k <= 2:
        raise ValueError("K must be bigger than two")
    else:
        ones = np.ones((np.size(x,0),1))
        x = np.hstack((ones,x))
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = tsize, random_state=5)
        Y_train = Y_train.reshape((np.size(Y_train,0),1))
        probability_matrix = np.zeros((len(X_test),k))
        prob_index = np.zeros((len(x),k))
        theta = np.zeros((np.size(X_test,1),k))
        target = np.zeros((len(Y_train),1))
        for t in range(len(Y_train)):
            target[t] = Y_train[t]
        for i in range(k):
            for w in range(len(Y_train)):
                if Y_train[w] == i:
                    target[w] = 1
                else:
                    target[w] = 0
            parameters = newton_step(X_train,target,15)
            theta[:,i] = parameters[:,0]
            prob = probability_vector(X_test,parameters)
            prob_final = probability_vector(x,parameters)
            probability_matrix[:,i] = prob[:,0]
            prob_index[:,i] = prob_final[:,0]
        result = np.zeros((len(probability_matrix),1))
        result_final = np.zeros((len(x),1))
        results_loss = np.zeros((len(probability_matrix),1))
        for i in range(len(probability_matrix)):
            results_loss[i,0] = probability_matrix[i,np.argmax(probability_matrix[i,:])]
            result[i,0] = np.argmax(probability_matrix[i,:])
        for i in range(len(prob_final)):
            result_final[i,0] = np.argmax(prob_index[i,:])
        for i in range(len(result)):
            counter = 0
            if np.absolute(result[i] - Y_test[i]) == 0:
                counter = counter 
            else:
                counter = counter + 1
        accuracy = np.round(((np.size(Y_test,0) - counter)/np.size(Y_test,0)) * 100)
        exp_loss = 0
        for i in range(len(result)):
            exp_loss += np.exp(-1 * Y_test[i] * results_loss[i] )
        return theta, accuracy, float(exp_loss), result_final
def acc_and_loss(xtest,ytest,parameters,k):
    """
    Calculates the accuracy and the huber loss of a given model w.r.t. the test dataset.
    Inputs:
        xtest: array
            input array of all points that constitute the input test dataset.
        ytest: array
            input array of all points that constitute the output test dataset.
        parameters: array
            input array of parameters/predictors to be used in the prediction algorithm.
        k: int
            input number of classes of the classification problem.
    Returns:
        accuracy: float
            outputs the approximate percentual accuracy of the model w.r.t. the test set.
        exp_loss" float
            outputs the approximate exponential loss of the model w.r.t. the test set.
        
    """
    ytest = ytest.reshape((-1,1))
    if k == 1:
        raise ValueError("k must be >= 2") 
    if k == 2:
        ones = np.ones((np.size(xtest,0),1))
        xtest = np.hstack((ones,xtest))
        prediction = np.round(probability_vector(xtest,parameters))
        counter = 0
        for i in range(len(ytest)):
            if np.absolute(ytest[i] - prediction[i]) == 0:
                counter = counter
            else:
                counter += 1
        accuracy = ((len(ytest) - counter)/len(ytest)) * 100
        exp_loss = 0
        prediction_loss = prediction
        for i in range(len(ytest)):
            exp_loss += np.exp(-1 * ytest[i] * prediction_loss[i])
        return np.round(accuracy,2), np.round(exp_loss,2), prediction
    if k >= 3:
        ones = np.ones((np.size(xtest,0),1))
        xtest = np.hstack((ones,xtest))
        probability_matrix = np.zeros((len(ytest),k))
        prediction = np.zeros((len(ytest),1))
        for j in range(k):
            prob = probability_vector(xtest,parameters[:,j])
            probability_matrix[:,j] = prob[:,0]
        for i in range(len(prediction)):
            prediction[i,0] = np.argmax(probability_matrix[i,:])
        results_loss = np.zeros((len(probability_matrix),1))
        for i in range(len(probability_matrix)):
            results_loss[i,0] = probability_matrix[i,np.argmax(probability_matrix[i,:])]
        counter = 0
        for i in range(len(ytest)):
            if np.absolute(ytest[i] - prediction[i]) == 0:
                counter = counter
            else:
                counter += 1
        accuracy = ((len(ytest) - counter)/len(ytest)) * 100
        exp_loss = 0
        for i in range(len(ytest)):
            exp_loss += np.exp(-1 * ytest[i] * results_loss[i] )
        return accuracy, float(exp_loss,2), prediction
class IRLSClassifier:
    """
    Iteratively Reweighted Least Squares algorithmn for classification, that can solve both binary and multiclass problems.
    Executes a given model based on user input.
    """
    def __init__(self,k,tsize = 0.2):
        """
        Initialize self.
        Inputs:
            k: int
                input the number k of classes associated with the dataset 
            tsize: float
                Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used in the validation set.
        """
        self.k = k 
        if self.k <= 1:
            raise ValueError("Insert a valida value for K")
        self.tsize = tsize 
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
        self.x = x 
        self.y = y 
        if self.k == 2:
            self.result = binary_classification_default(self.x,self.y,self.tsize)
        elif self.k >= 3 : 
            self.result = one_vs_all_default(self.x,self.y,self.k,self.tsize)
        else:
            raise ValueError("Invalid value of k for classification.")
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
        self.input = x 
        ones = np.ones((len(self.input),1))
        self.input = np.hstack((ones,self.input))
        if self.k == 2:
            self.predict = np.round(probability_vector(self.input,self.result[0]))
            return self.predict
        else:
            probability_matrix = np.zeros((len(self.input),self.k))
            self.predict = np.zeros((len(self.input),1))
            for j in range(self.k):
                prob = probability_vector(self.input,self.result[0][:,j])
                probability_matrix[:,j] = prob[:,0]
            for i in range(len(self.predict)):
                self.predict[i,0] = np.argmax(probability_matrix[i,:])   
            return self.predict 
    def parameters(self):
        """
        Gives the parameters calculated by the classification function.
        Returns:
            self.result[0]: array
                outputs the array of parameters calculated by the classification function.
        """
        return self.result[0]
    def val_error(self,etype = 'acc'):
        """
        Gives two different types of error calculated on the validation set,i.e., calculates the expected test error using the validation set.  
        Inputs: 
            etype: string
                input string that identifies the type of error to be calculated. etype can be: 'acc'(Accuracy) or 'exp'(Exponential Loss); default is set to 'acc'.
        Returns:
            self.result[1]: float
                outputs the accuracy of the model calculated on the validation set.
            self.result[2]: float:
                outputs the Exponential Loss of the model calculated on the validation set.
        """
        self.etype = etype
        if self.etype == 'acc':
            return float(self.result[1])
        elif self.etype == 'exp':
            return float(self.result[2])
        else:
            print("Please insert a valid error type.")
            return ValueError
