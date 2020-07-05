

# # basicMLpy Module


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
			Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used in the test set.
	Returns:
	    theta: array
	        outputs array of predictors/parameters calculated by the algorithm.
	    accuracy: float
	        outputs the approximate percentual accuracy of the model, counting each misclassification on the validation set and calculating the final score.    
		exp_loss: float
            outputs the approximate exponential loss of the model on the validation set.
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
	return theta, accuracy, np.round(exp_loss,2), np.round(prediction_final)
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
			Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used in the test set.
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
		print("K must be bigger than two")
		return ValueError 
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
		for n in range(len(probability_matrix)):
			result[n,0] = np.argmax(probability_matrix[n,:])
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
		return theta, accuracy, np.round(exp_loss,2), result_final
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
        k: integer
            input number of classes of the classification problem.
    Returns:
        accuracy: float
            outputs the approximate percentual accuracy of the model w.r.t. the test set.
        exp_loss" float
            outputs the approximate exponential loss of the model w.r.t. the test set.
        
	"""
	ytest = ytest.reshape((-1,1))
	if k == 1:
		print("k must be >= 2")
		return ValueError 
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
		return np.round(accuracy,2), np.round(exp_loss,2)
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
		return np.round(accuracy,2), np.round(exp_loss,2)
class classification_fit:
	"""
	Class of two different types of classification fits for a given dataset.
	Executes a given model based on user input.
		Inputs:
			x: array
				input array of input points to be used as training set, without the intercept(raw data).
			y: array
				input array of output points, usually a column vector with same number of rows as x.
			k: int
				input the number k of classes associated with the dataset 
			tsize: float
				Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used in the test set.
		Returns:
			self.bt: array
				outputs the array of parameters/predictors calculated by the binary model.
			self.ba: float
				outputs the accuracy of the binary model.
			self.bel: float
			    outputs the approximate exponential loss of the model on the validation set.
			self.bpredict: array
			    outputs the array of predictions over all inputs using the calculated parameters.
			self.kt: array 
				outputs the array of parameters/predictors calculated by the one-vs-all model.
			self.ka: float
				outputs the accuracy of the one-vs-all model. 
			self.kel: float
			    outputs the approximate exponential loss of the model on the validation set.
			self.kpredict:
			    outputs the array of predictions over all inputs using the calculated parameters.
	"""
	def __init__(self,x,y,k,tsize = 0.2):
		if k == 2:
			self.bt = binary_classification_default(x,y,tsize)[0]
			self.ba = binary_classification_default(x,y,tsize)[1]
			self.bel = binary_classification_default(x,y,tsize)[2]
			self.bpredict = binary_classification_default(x,y,tsize)[3]
		else:
			self.kt = one_vs_all_default(x,y,k,tsize)[0]
			self.ka = one_vs_all_default(x,y,k,tsize)[1]
			self.kel = one_vs_all_default(x,y,k,tsize)[2]
			self.kpredict = one_vs_all_default(x,y,k,tsize)[3]
#REGRESSION#
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
	errors_sum = sum(errors)
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
			Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used in the test set.
	Returns:
		theta: array
			outputs the array of predictors for the regression model.
		mse: float
			outputs the approximate Mean Squared Error found by the regression model on the validation set. mse is rounded up to 2 decimal cases.
		prediction: array
			outputs the array of predictions over all inputs using the calculated parameters.
		huber_loss: float
			outputs the approximate huber loss for the validation set.
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
	return theta, np.round(mse,2), prediction, np.round(huber_loss,2)
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
			Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used in the test set.
	Returns:
		theta: array
			outputs the array of predictors for the regression model.
		mse: float
			outputs the approximate Mean Squared Error found by the regression model for the validation set. mse is rounded up to 2 decimal cases.
		prediction: array
			outputs the array of predictions over all inputs using the calculated parameters.
		huber_loss: float
			outputs the approximate huber loss for the validation set.
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
		return theta,np.round(mse,2), prediction, np.round(huber_loss,2)
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
		return theta,np.round(mse,2), prediction, np.round(huber_loss,2)
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
			Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used in the test set.
	Returns:
		theta: array
			outputs the array of predictors for the regression model.
		mse: float
			outputs the approximate Mean Squared Error found by the regression model. mse is rounded up to 2 decimal cases.
		prediction: array
			outputs the array of predictions over all inputs using the calculated parameters.
		huber_loss: float
			outputs the approximate huber loss for the validation set.
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
	return theta, np.round(mse,2), prediction, np.round(huber_loss)
def mse_and_huber(xtest,ytest,parameters):
    """
    Calculates the Mean Squared Error and the Huber Loss of a given model w.r.t. the test dataset.
    Inputs:
        xtest: array
            input array of all points that constitute the input test dataset, without the intercept(raw data).
        ytest: array
            input array of all points that constitute the output test dataset.
        parameters: array
            input array of parameters/predictors to be used in the prediction algorithm.
    Returns:
        mse: float
            outputs the approximate Mean Squared Error w.r.t. the test set.
        huber_loss: float
            outputs the approximate Huber Loss w.r.t. the test set.
            
        
"""
    ones = np.ones((len(xtest),1))
    xtest = np.hstack((ones,xtest))
    prediction = xtest @ parameters
    error = sum(np.square(ytest - prediction))
    mse = error/len(ytest)
    huber_loss = 0
    delta = 0.5
    for i in range(len(xtest)):
        if np.absolute(ytest[i] - prediction[i] ) <= delta:
            huber_loss += np.square(ytest[i] - prediction[i])
        else:
            huber_loss += 2 * delta * np.absolute(ytest[i] - prediction[i]) - delta**2
    return np.round(mse,2), np.round(huber_loss,2)
class regression_fit:
	"""
	Class of three different regression models.
	Executes a given model based on user input.
	Inputs:
		x: array
			input array of input points, without the intercept(raw data).
		y: array
			input array of output points, usually a column vector.
		btype: string
			input string that identifies the type of basis expansion; btype can be: None, sqrt, poly; default is set to None.
		reg_lambda: float
			input string specifies the regularization constant to be used in the ridge regresssion; default is set to None.
		tsize: float
			Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used in the test set.
	Returns:
		self.lt: array
			outputs the array of parameters/predictors calculated by the standard linear model.
		self.le: float
			outputs the approximate Mean Squared Error calculated by the standard linear model.
		self.lhl: float
			outputs the approximate huber loss for the validation set.
		self.lpredict: array
			outputs the array of predictions over all inputs using the calculated parameters.
		self.rt: array
			outputs the array of parameters/predictors calculated by the ridge linear model.
		self.re: float
			outputs the approximate Mean Squared Error calculated by the ridge linear model.
		self.rhl: float
			outputs the approximate huber loss for the validation set.
		self.rpredict: array
			outputs the array of predictions over all inputs using the calculated parameters.
		self.bet: array
			outputs the array of parameters/predictors calculated by the basis expansion model.
		self.bee: float
			outputs the approximate Mean Squared Error calculated by the basis expansion model.
		self.behl: float
			outputs the approximate huber loss for the validation set.
		self.bepredict: array
			outputs the array of predictions over all inputs using the calculated parameters.
    """
	def __init__(self,x,y,btype = None,reg_lambda = None, tsize = 0.2):
		if btype == None:
			if reg_lambda == None:            
				self.lt = linear_regression(x,y,tsize)[0]
				self.le = linear_regression(x,y,tsize)[1]
				self.lpredict = linear_regression(x,y,tsize)[2]
				self.lhl = linear_regression(x,y,tsize)[3]
			else:
				self.rt = ridge_regression(x,y,reg_lambda,tsize)[0]
				self.re = ridge_regression(x,y,reg_lambda,tsize)[1]
				self.rpredict = ridge_regression(x,y,reg_lambda,tsize)[2]
				self.rhl = ridge_regression(x,y,reg_lambda,tsize)[3]
		else:
			self.bet = basis_expansion(x,y,btype,tsize)[0]
			self.bee = basis_expansion(x,y,btype,tsize)[1]
			self.bepredict = basis_expansion(x,y,btype,tsize)[2]
			self.behl = basis_expansion(x,y,btype,tsize)[3]


