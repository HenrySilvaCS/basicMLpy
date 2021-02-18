# # basicMLpy.loss_functions module
import numpy as np 

def check_inputs_shape(arr1,arr2):
	"""
	Checks if the dimensions of two arrays match and prints a warning if the dont.
	Inputs:
	    arr1: array
	        input array whose dimension is to be checked
	    arr2: array
	        input array whose dimension is to be checked      

	"""
	if (len(arr1.shape) != (len(arr2.shape))):
		print("WARNING! The shape of the prediction array is different to that of the target array, this may lead to errors.")
		print("Try passing arrays with equal shapes!")

def mse(prediction,target):
	"""
	Calculates the Mean Squared Error between two arrays
	Inputs:
	    prediction: array of shape (N,*)
	        input array of predictions
	    target: array of shape (N,*), same as prediction
	        input array of targets
	Outputs:
	    error: float or array of floats
	        outputs the mean squared error between the two arrays

	"""
	check_inputs_shape(prediction,target)
	error = np.sum(np.square(target - prediction))
	return (error/len(target))

def mae(prediction,target):
	"""
	Calculates the Mean Absolute Error between two arrays
	Inputs:
	    prediction: array of shape (N,*)
	        input array of predictions
	    target: array of shape (N,*), same as prediction
	        input array of targets
	Outputs:
	    error: float or array of floats
	        outputs the mean squared absolute between the two arrays
	"""
	check_inputs_shape(prediction,target)
	error = np.sum(np.abs(target - prediction))
	return (error/len(target))

def huber_loss(prediction,target,delta=0.5):
	"""
	Calculates the Huber Loss(also referred as Smooth L1 Loss) between two arrays
	Inputs:
	    prediction: array
	        input array of predictions
	    target: array
	        input array of targets
	    delta: float
	        input delta constant for the huber loss. default = 0.5
	Outputs:
	    huber_loss: float
	        outputs the huber loss between the two arrays
	"""
	check_inputs_shape(prediction,target)
	huber_loss = 0 
	for i in range(len(target)):
		if np.absolute(target[i] - prediction[i] ) <= delta:
			huber_loss += 0.5 * np.square(target[i] - prediction[i])
		else:
			huber_loss += (delta * np.absolute(target[i] - prediction[i])) -(0.5 * (delta**2))
	return huber_loss

def exponential_loss(prediction,target):
	"""
	Calculates the Exponential Loss between two arrays
	Inputs:
	    prediction: array
	        input array of predictions
	    target: array
	        input array of targets
	Outputs:
	    exp_loss: float
	        outputs the exponential loss between the two arrays
	"""
	check_inputs_shape(prediction,target)
	return np.mean(np.exp(- prediction * target))

def standard_accuracy(prediction,target,round_prediction=False):
	"""
	Calculates the accuracy of the predictions, given a target
	Inputs:
	    prediction: array
	        input array of predictions
	    target: array
	        input array of targets
	Output
	"""
	check_inputs_shape(prediction,target)
	counter = 0
	if(round_prediction):
		rounded_prediction = np.round(prediction)
		for i in range(len(prediction)):
			if np.absolute(rounded_prediction[i] - target[i]) == 0:
				counter = counter 
			else:
				counter += 1
		accuracy = ((np.size(target,0) - counter)/np.size(target,0))
		return accuracy
	else:
		for i in range(len(prediction)):
			if np.absolute(prediction[i] - target[i]) == 0:
				counter = counter 
			else:
				counter += 1
		accuracy = ((np.size(target,0) - counter)/np.size(target,0))
		return accuracy
		
def misclassification(y,prediction):
    """
    Calculates the number of misclassifications between the prediction and the output.
    Inputs:
        y: int
            Input a single value of the output variable y.
        prediction: int
            Input a single valut of the predicted output.
    Returns:
        misclassifications: array
            Output 1 if the values do not match and 0 if they do.
    """
    y=y.reshape((-1,1))
    prediction = prediction.reshape((-1,1))
    misclassifications = 1*(y != prediction)
    return misclassifications