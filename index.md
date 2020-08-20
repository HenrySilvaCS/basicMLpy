[Home](https://henrysilvacs.github.io/basicMLpy/)  | [Install](https://henrysilvacs.github.io/basicMLpy/install) | [User Guide](https://henrysilvacs.github.io/basicMLpy/user_guide) | [Coming up next](https://henrysilvacs.github.io/basicMLpy/coming_up_next) | [About the author](https://henrysilvacs.github.io/basicMLpy/about)
# Introduction to the package
basicMLpy is a python package focused on implementing machine learning algorithms for supervised learning tasks. It currently has 5 fully functional modules, that provide implementations of various models for supervised learning, and also many functions for model selection and error evaluation.
## Package features
The package currently contains five different modules. Their functionalities are described below:


### basicMLpy.regression module contains the following functionalities:
* Linear Regression 
* Ridge Regression 
* Basis expanded regression, that allows for nonlinear models 
* Error evaluation through Mean Squared Error and Huber Loss


### basicMLpy.classification module contains the following functionalities:
* Multiclass classification through the IRLS(Iteratively Reweighted Least Squares) algorithm
* Error evaluation through accuracy and exponential loss


### basicMLpy.nearest_neighbors module contains the following functionalities:
* An implementation of the K-Nearest Neighbors algorithm, that can fit both classification and regression problems


### basicMLpy.cross_validation module contains the following functionalities:
* A Cross-Validation algorithm for the functions presented by the basicMLpy package
* Functions for model selection


### basicMLpy.ensemble module contains the following functionalities:
* An implementation of the Random Forests algorithm for regression and classification
* An implementation of the AdaBoost algorithm for classification
* An implementation of the Gradient Boosting algorithm for regression

  
