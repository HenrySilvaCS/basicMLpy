# # basicMLpy.regression module
import numpy as np 
from basicMLpy.utils import check_for_intercept
from scipy import linalg


def qr_regression(x,y):
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
	x = check_for_intercept(x)
	q, r = np.linalg.qr(x) 
	b = q.T @ y
	theta = linalg.solve_triangular(r,b)
	return theta

def ridge_regression(x,y,reg_const):
	"""
	Fits a ridge linear regression model on a given dataset.
	Inputs:
		x: array
			input array of input points to be used as training set, with the intercept.
		y: array
			input array of output points, usually a column vector with same number of rows as x.
		reg_const: float
			input the value for the penalizing constant(lambda) used by the ridge algorithm.
	Returns:
		theta: array
			outputs the array of predictors for the regression model.
	"""
	x = check_for_intercept(x)
	identity = np.eye(np.size(x,1))
	theta = np.linalg.inv(x.T @ x + reg_const * identity) @ x.T @ y 
	return theta

class LinearRegression:
	"""
	Class of the Linear Regression model.
	Methods:
		fit(x,y) -> Performs the linear regression algorithm on the training set(x,y).
		predict(x) -> Predict value for X.
		parameters() -> Returns the calculated parameters for the linear model.
	"""
	def fit(self,x,y):
		"""
		Fits one of the regression models on the dataset.
		Inputs:
			x: array
				input array of input points.
			y: array
				input array of output points.
		Functionality:
			stores all information from a given function into self.result. this statement will be used later in other functions.
		"""
		self.parameters = qr_regression(x,y)
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
		x = check_for_intercept(x)
		predict = x @ self.parameters
		return predict
	def get_parameters(self):
		"""
		Gives the parameters calculated by the regression function.
		Returns:
			self.result[0]: array
				outputs the array of parameters calculated by the regression function.
		"""
		return self.parameters

class RidgeRegression(LinearRegression):
	"""
	Class of the ridge regression model. Inherits from the LinearRegression class.
	Methods:
		fit(x,y) -> Performs the linear regression algorithm on the training set(x,y).
		predict(x) -> Predict value for X.
		parameters() -> Returns the calculated parameters for the linear model.
	"""
	def __init__(self,reg_lambda):
		self.reg_lambda = reg_lambda

	def fit(self,x,y):
		"""
		Fits one of the regression models on the dataset.
		Inputs:
			x: array
				input array of input points, without the intercept(raw data).
			y: array
				input array of output points, usually a column vector.
			reg_lambda: float
				input value that specifies the regularization constant to be used in the ridge regresssion;
		"""
		self.parameters = ridge_regression(x,y,self.reg_lambda)

