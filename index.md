[Home](https://henrysilvacs.github.io/basicMLpy/)  | [User Guide](https://henrysilvacs.github.io/basicMLpy/user_guide)
# Introduction to the package
basicMLpy is a python package focused on implementing machine learning algorithms for supervised learning tasks. It currently has 5 fully functional modules, that provide implementations of various models for supervised learning, and also many functions for model selection and error evaluation.
## About the author
I started this as a personnal project during my scientific research on machine learning and data science at the Federal University of Minas Gerais(UFMG), as a way to practice and get experience with the machine learning techniques that I am learning. I'm currently at the first semester of a 10-semester long Computer Science course, so I'm still new to coding and machine learning in general. With that in mind, this package isn't by any means meant to be on the same level of other big python packages for machine learning, such as scikitlearn or xgboost, it is built to at least have a comparable performance to the aforementioned packages when used to fit small to medium sized datasets. The package's functionalities are also easy-to-use and pretty intuitive, so it also provides a welcoming environment for newcomers.
## Installation
basicMLpy can be installed by downloading the latest official release. Currently the package is only available through the pip package.<br />
To install the latest version of basicMLpy, run the following command:<br />
```
pip install -i https://test.pypi.org/simple/ basicMLpy
```
If you want to install a specific version, run the following:<br />
```
pip install -i https://test.pypi.org/simple/ basicMLpy==*version*
```
The source code for the latest version of the package is available [Here](https://github.com/HenrySilvaCS/basicMLpy)<br />
The source code for all the versions of the package is available [Here](https://test.pypi.org/project/basicMLpy/#history)
## Package features
The package currently contains five different modules. Their functionalities are described below:
### | &nbsp;basicMLpy.classification module contains the following functionalities:
* Multiclass classification through the IRLS(Iteratively Reweighted Least Squares) algorithm
* Error evaluation through accuracy and exponential loss
### | basicMLpy.nearest_neighbors module contains the following functionalities:
* An implementation of the K-Nearest Neighbors algorithm, that can fit both classification and regression problems
### | basicMLpy.cross_validation module contains the following functionalities:
* A Cross-Validation algorithm for the functions presented by the basicMLpy package
* Functions for model selection<br />
### | basicMLpy.ensemble module contains the following functionalities:
* An implementation of the Random Forests algorithm for regression and classification
* An implementation of the AdaBoost algorithm for classification
* An implementation of the Gradient Boosting algorithm for regression<br />

  
